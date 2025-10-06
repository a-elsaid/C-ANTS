from contextlib import nullcontext
from sklearn.cluster import DBSCAN
import numpy as np
from point import Point
from util import get_center_of_mass, function_names
import loguru
import graphviz as gv
from time import time
import sys
import torch
import matplotlib.pyplot as plt
from typing import Dict
import pickle
from tqdm import tqdm
from rnn_node import *
from graph_node import *

# import ipdb

def sigmoid(x):
    if type(x) == torch.Tensor:
        return torch.sigmoid(x)
    x = np.clip(x, -500, 500)  # Avoid overflow
    return 1 / (1 + np.exp(-x))
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) if x.ndim == 1 else e_x / e_x.sum(axis=1, keepdims=True)

logger = loguru.logger
logger.remove()  # Remove default logger
logger.add(sys.stdout, level="INFO")

class Structure:
    counter = 0
    def __init__(   
            self, 
            ants_paths,
            cants_ants: str,
            Node_Class,
            lags,
            eps,
            min_samples,
            use_torch,
            colony_id,
            cost_type="mse",
    ):
        self.id = Structure.counter
        Structure.counter += 1
        self.use_torch = use_torch
        self.Node_Class = Node_Class
        self.colony_id = colony_id
        self.eps = eps
        self.min_samples = min_samples
        self.ants_paths = ants_paths
        self.lags = lags
        self.nodes = {}         # TODO: hide this from the user (after testing)
        self.in_nodes = []      # TODO: hide this from the user (after testing)
        self.out_nodes = []     # TODO: hide this from the user (after testing)
        self.added_points = []
        self.added_in_points = []

        self.cost_type = cost_type
        # Create nodes
        '''
        Creating nodes of the structure from the ants paths
        Works for both CANTS and ANTS
        For CANTS: Merge nodes in the same cluster, clean the structure, fix lags -> Next step: if CANTS ↓
        '''
        self.create_nodes() 
        if cants_ants=="CANTS":
            # Merge nodes in the same cluster
            self.merge_nodes()
            # Clean the structure
            self.clean_structure()
            # Fix lags
            self.add_lagged_inputs()
        
            
        # ipdb.set_trace()

    def get_added_points(self,):
        return self.added_points
    
    def get_added_in_points(self,):
        return self.added_in_points

    
    def get_edges(self,):
        return [edge for node in self.nodes.values() for edge in node.outbound_edges.values()]

    def create_nodes(self, ):
        nodes_of_input_points = {}
        nodes_of_output_points = {}
        nodes = {}
        def add_node(curr_point, next_point, curr_node=None):
            if curr_node is None:
                # Check if the node already exists
                curr_node = nodes_of_input_points.get(
                                                    curr_point.name, 
                                                    self.Node_Class(type=curr_point.get_node_type(), point=curr_point, lags=self.lags, use_torch=self.use_torch)
                                                    )  
                nodes_of_input_points[curr_point.name] = curr_node
            
            logger.debug(f"Node({curr_node.id}) - Point({curr_point.get_id()}) - Lag({curr_node.lag}) - Point_Z({curr_point.get_z()}), Node_Z({curr_node.z})")
            
            if next_point.get_node_type() == 2:
                # Check if the node already exists
                next_node = nodes_of_output_points.get(
                                                    next_point.get_id(), 
                                                    self.Node_Class(type=next_point.get_node_type(), point=next_point, lags=self.lags, use_torch=self.use_torch)
                                                    )
                nodes_of_output_points[next_point.get_id()] = next_node
            elif next_point.get_node_type() == 0:
                # Check if the node already exists
                next_node = nodes.get(
                                                    next_point.get_id(), 
                                                    self.Node_Class(type=next_point.get_node_type(), point=next_point, lags=self.lags, use_torch=self.use_torch)
                                    )
                nodes[next_point.get_id()] = next_node   # Add All node
            else:
                print(f"Unknown node type or Input node: {next_point.get_node_type()}")
                exit(1)
            
            # print(f"Point({curr_point.get_id()}) Node({curr_node.id}) ->", end=' ')

            

            curr_node.add_edge(next_node)                        # adding edge between nodes   
            return next_node

        # Create nodes
        for p, path in enumerate(self.ants_paths):
            logger.debug(f"Creating nodes for Path: {p}")
            next_node = None
            for i in range(len(path)-1):
                next_node = add_node(
                                            curr_point=path[i], 
                                            next_point=path[i+1], 
                                            curr_node=next_node,
                                        )
            # print(f"Point({path[-1].get_id()}) Node({next_node.id})")
            # print()
            
        self.nodes = {node.id:node for node in {**nodes_of_input_points, **nodes, **nodes_of_output_points}.values()}
        self.in_nodes = list(nodes_of_input_points.values())
        self.out_nodes = list(nodes_of_output_points.values())



    def remove_node(self, node_id):
        # Remove node & all its references from the structure
        logger.debug(f"Removing Node({node_id})")
        for inbound in self.nodes[node_id].inbound_edges.values():
            if inbound.source.id in self.nodes:
                del self.nodes[inbound.source.id].outbound_edges[node_id]
        for outbound in self.nodes[node_id].outbound_edges.values():
            if outbound.target.id in self.nodes:
                del self.nodes[outbound.target.id].inbound_edges[node_id]
        del self.nodes[node_id]

    def remove_edge(self, source_id, target_id):
        # Remove edge from the structure
        del self.nodes[source_id].outbound_edges[target_id]
        del self.nodes[target_id].inbound_edges[source_id]


    def merge_nodes(self,):
        # Merge nodes in the same cluster
        points_coordinates = np.array([node.point.coordinates() for node in self.nodes.values() if node.point.get_node_type() == 0])
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_coordinates)
        labels = db.labels_
        labels_set = set(labels)
        
        nodes = np.array([n for n in self.nodes.values() if n.point.get_node_type() == 0])  
        new_nodes = []
        del_nodes = []
        for label in labels_set:
            if label == -1:
                continue
            cluster_nodes = nodes[labels == label]
            logger.debug(f"Merging nodes in Cluster {label} with {len(cluster_nodes)} nodes")
            center_of_mass_point = get_center_of_mass(
                                                        [node.point for node in cluster_nodes 
                                                            if node.point.get_node_type() == 0
                                                        ]
                                                       )
            new_cluster_point = Point(*center_of_mass_point, type=0)
            # self.space.points.append(new_cluster_point)   # Adding new points in the colony to decouple the structure from the colony
            self.added_points.append(new_cluster_point)
            new_cluster_node = self.Node_Class(point=new_cluster_point, lags=self.lags, use_torch=self.use_torch)
            logger.debug(f"\tNew cluster Point({new_cluster_point.get_id()}) - Node({new_cluster_node.id})")
            new_nodes.append(new_cluster_node)

            # Reassign edges to new node
            logger.debug(f"Reassigning functions and edges for Cluster({label})")
            for node in cluster_nodes:
                logger.debug(f"\tReassigning functions from Node({node.id}) \
                        to new cluster Node({new_cluster_node.id})")
                new_cluster_node.add_functions(node.functions)
                logger.debug(f"\tReassigning edges from Node({node.id}) \
                        to new cluster Node({new_cluster_node.id})")
                for edge in node.outbound_edges.values():
                    edge.source = new_cluster_node
                    new_cluster_node.outbound_edges[edge.target.id] = edge
                    logger.debug(f"\t\tRemoving edge between Node({node.id}) and Node({edge.target.id}) \
                                -- Replace with edge between Node({new_cluster_node.id}) and Node({edge.target.id})")
                    del edge.target.inbound_edges[node.id]
                    edge.target.inbound_edges[new_cluster_node.id] = edge
                for edge in node.inbound_edges.values():
                    edge.target = new_cluster_node
                    new_cluster_node.inbound_edges[edge.source.id] = edge
                    logger.debug(f"\t\tRemoving edge between Node({edge.source.id}) and Node({node.id}) \
                                -- Replace with edge between Node({edge.source.id}) and Node({new_cluster_node.id})")
                    del edge.source.outbound_edges[node.id]
                    edge.source.outbound_edges[new_cluster_node.id] = edge
                logger.debug(f"\tRemoving Node({node.id})")

                del_nodes.append(node.id)
        for node_id in del_nodes:
            del self.nodes[node_id]
        for node in new_nodes:
            self.nodes[node.id] = node


    def detect_and_remove_cycles(self,):
        visited = set() # Nodes that have been visited
        stack = set()   # Nodes that are currently being visited

        def dfs(node, prev_node=None):
            visited.add(node)
            stack.add(node)
            for edge in node.outbound_edges.values():
                if edge.target not in visited:
                    if dfs(edge.target, node):
                        return True
                elif edge.target in stack:
                    # Cycle detected: Remove edge that close the cycle
                    logger.debug(f"Cycle detected: Removing Edge between Node({node.id}) and Node({edge.target.id})")
                    self.remove_edge(node.id, edge.target.id)
                    return True

            stack.remove(node)
            return False

        # Run DFS on all nodes to detect cycles
        for node in self.nodes.values():
            if node not in visited:
                cycle_detected = dfs(node)
                while cycle_detected:
                    visited = set()
                    stack = set()
                    cycle_detected = dfs(node)

    def remove_dead_ends(self,):
        # Step 1: Remove dead ends (nodes with no outbound edges or no inbound edges except for input and output nodes)
        dead_ends = [node for node in self.nodes.values() 
                            if (
                                len(node.outbound_edges) == 0 or 
                                len(node.inbound_edges)  == 0
                               ) and
                               node.type == 0
                    ]
        for node in dead_ends:
            self.remove_node(node.id)

        # Step 2: Remove nodes not reachable from input nodes
        
        def dfs(node, from_in):
            if from_in:
                edge_list = node.outbound_edges
                reachable = reachable_from_in
            else:
                edge_list = node.inbound_edges
                reachable = reachable_from_out
            reachable.add(node)
            for edge in edge_list.values():
                if from_in:
                    if edge.target not in reachable:
                        dfs(edge.target, from_in)
                else:
                    if edge.source not in reachable:
                        dfs(edge.source, from_in)
        
        # Run DFS from input nodes
        reachable_from_in = set()
        for node in self.in_nodes:
            dfs(node, from_in=True)

        # Run DFS from output nodes
        reachable_from_out = set()
        for node in self.out_nodes:
            dfs(node, from_in=False)

        # Remove nodes not reachable from input nodes
        nodes_ids = list(self.nodes.keys())
        unreachable_from_in  = [node.id for node in self.nodes.values() if node not in reachable_from_in]
        unreachable_from_out = [node.id for node in self.nodes.values() if node not in reachable_from_out]
        unreachable = set(unreachable_from_in + unreachable_from_out)
        logger.trace(f"Unreachable nodes: {[node_id for node_id in unreachable]}")
        
        for node_id in unreachable:
            logger.debug(f"Removing Node({node_id}) as it is not reachable from input nodes")
            self.remove_node(node_id)

        # Step 3: Remove unconnected input and output nodes
        node_to_remove = []
        for node in self.in_nodes:
            if node.id not in self.nodes:
                node_to_remove.append(node)
        for node in node_to_remove:
            self.in_nodes.remove(node)

        node_to_remove = []             
        for node in self.out_nodes:
            if node.id not in self.nodes:
                node_to_remove.append(node)
        for node in node_to_remove:
                self.out_nodes.remove(node)

    def clean_structure(self,):
        self.detect_and_remove_cycles() # Detect and remove cycles
        self.remove_dead_ends()         # Remove dead ends


    def add_lagged_inputs(self,):
        added_points = []
        added_nodes = []
        visited = set()

        def thrust_forward(node, prev_node, in_node):
            if node in visited:
                return
            visited.add(node)
            if node.z<prev_node.z:
                if node.z<=in_node.z:
                    new_input_lag_point = Point(
                                                    x = in_node.point.get_x(), 
                                                    y = 0, 
                                                    z = node.z, 
                                                    f = in_node.point.get_f(), 
                                                    type=1,
                                                )
                    logger.trace(f"LAG POINT: Adding point: Node({new_input_lag_point.get_id()})")
                    input_index = round(new_input_lag_point.get_x()*(len(self.space.input_names)-1))
                    new_input_lag_point.name = self.space.input_names[input_index]
                    new_input_lag_point.name_idx = input_index
                    logger.trace(f"LAG NODE: Adding node: Node({new_input_lag_point.get_id()})")
                    new_input_lag_node = self.Node_Class(point=new_input_lag_point, type=1, lags=self.lags, use_torch=self.use_torch)
                    new_input_lag_node.add_edge(node)
                    # self.space.input_points.append(new_input_lag_point)   #adding new points in colony to decouple the structure from the colony
                    self.added_in_points.append(new_input_lag_point)
                    # added_points.append(new_input_lag_point)              #adding new points in colony to decouple the structure from the colony
                    self.added_points.append(new_input_lag_point)
                    added_nodes.append(new_input_lag_node)
                else:
                    logger.trace(f"LAG EDGE: Adding edge between Node({in_node.id}) and Node({node.id}), Type: {in_node.type}")
                    in_node.add_edge(node)

                    

            edges = list(node.outbound_edges.values())
            for edge in edges:
                thrust_forward(edge.target, node, in_node)

        
        #iterate over input nodes to start there and reach output nodes
        for in_node in self.in_nodes:
            edges = list(in_node.outbound_edges.values())
            for edge in edges:
                thrust_forward(edge.target, in_node, in_node)

        # for point in added_points:
        #     self.space.points.append(point) #adding new points in colony to decouple the structure from the colony
        for node in added_nodes:
            self.nodes[node.id] = node
            self.in_nodes.append(node)
    


    def mse(self, y_true, y_pred, prt=False):
        # mse = np.array(y_true) - np.array(y_pred)
        err = y_true - y_pred
        if prt: print(f"Prediction: {y_pred.item():.7f}, GT: {y_true.item():.7f}, Err: {err.item():.7f}")
        mse = (err*err)/2
        d_mse = err
        logger.trace(f"Mean Squared Error: {mse} -- Derivative: {d_mse}")
        return mse, d_mse

    def bicross_entropy(self, y_true, y_pred, prt=False):
        """
        Binary Cross-Entropy loss (for binary or multi-label classification).
        Works with both torch.Tensor and numpy.ndarray.

        Parameters:
            y_true: binary ground truth (0 or 1), tensor or array
            y_pred: sigmoid output, same shape as y_true, in (0, 1)
        Returns:
            bce: scalar binary cross-entropy loss
            d_bce: gradient w.r.t. prediction
        """

        # Detect backend
        if isinstance(y_true, torch.Tensor):
            err = y_true - y_pred
            if prt: print(f"[Torch] Prediction: {y_pred.item():.7f}, GT: {y_true.item():.7f}, Err: {err.item():.7f}")
            bce = -torch.mean(y_true * torch.log(y_pred + 1e-8) + (1 - y_true) * torch.log(1 - y_pred + 1e-8))
            d_bce = err / (y_pred * (1 - y_pred) + 1e-8)

        elif isinstance(y_true, (np.ndarray, int, float)) and isinstance(y_pred, (np.ndarray, int, float)):
            y_true = np.array(y_true, dtype=np.float32)
            y_pred = np.array(y_pred, dtype=np.float32)
            err = y_true - y_pred
            if prt: print(f"[NumPy/Scalar] Prediction: {float(y_pred):.7f}, GT: {float(y_true):.7f}, Err: {float(err):.7f}")
            bce = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
            d_bce = err / (y_pred * (1 - y_pred) + 1e-8)

        logger.trace(f"Bicross Entropy: {bce} -- Derivative: {d_bce}")
        return bce, d_bce

    def cross_entropy(self, y_true, y_pred, prt=False):
        """
        Compute cross-entropy for one-hot targets.
        Works with either torch tensors or numpy arrays.
        Assumes y_true is a one-hot encoded vector (Tensor or NumPy array).
        Assumes y_pred is a softmaxed probability distribution.
        
        Parameters:
            y_true: one-hot encoded class label (tensor or np.array)
            y_pred: softmaxed output vector (tensor or np.array)
        Returns:
            ce: scalar cross-entropy loss
            d_ce: gradient w.r.t. y_pred (softmax output)
        """
        if isinstance(y_true, torch.Tensor): # torch tensor
            err = y_true - y_pred
            ce = -torch.sum(y_true * torch.log(y_pred + 1e-8))
            d_ce = y_pred - y_true
            if prt:
                pred_class = torch.argmax(y_pred).item()
                true_class = torch.argmax(y_true).item()
                print(f"[Torch] Pred: {pred_class}, GT: {true_class}, Softmax: {y_pred.tolist()}")
        else:
            err = y_true - y_pred
            ce = -np.sum(y_true * np.log(y_pred + 1e-8))
            d_ce = y_pred - y_true
            if prt:
                pred_class = int(np.argmax(y_pred))
                true_class = int(np.argmax(y_true))
                print(f"[NumPy] Pred: {pred_class}, GT: {true_class}, Softmax: {y_pred.tolist()}")

        logger.trace(f"Cross Entropy: {ce} -- Derivative: {d_ce}")
        return ce, d_ce
    
    def evaluate(
                    self, 
                    data, 
                    cal_gradient=True, 
                    cost_type="mse", 
                    num_epochs=1, 
                    thread_id=None, 
                    active_inference=False,
                    lr=0.001,
                    lr_discount = 0.98,
                    lr_step = 50,
                    bnn_kl_param=.01,
                ):
        if self.use_torch:
            input = torch.tensor(data.train_input, requires_grad=True, dtype=torch.float32)
            target = torch.tensor(data.train_output, requires_grad=True, dtype=torch.float32)
        else:
            input = data.train_input
            target = data.train_output
        
        thd = f" Thread({thread_id})" if thread_id is not None else " "
            
        # for epoch in tqdm(range(1,num_epochs+1), total=num_epochs, leave=False, desc=f"Colony({self.colony_id}){thd}:: Training"):
        for epoch in range(1,num_epochs+1):
            if epoch % lr_step==0:
               lr*=lr_discount
            time_start = time()
            preds, errors, d_errors = self.single_thrust(
                                                            input, 
                                                            target, 
                                                            prt=False, 
                                                            cal_gradient=cal_gradient, 
                                                            lr=lr,
                                                        )

            train_errors = torch.stack(errors) if self.use_torch else np.array(errors)
            train_d_errors = torch.stack(d_errors) if self.use_torch else np.array(d_errors)
            train_preds = torch.stack(preds) if self.use_torch else np.array(preds)
            
            mean_train_err = torch.mean(train_errors) if self.use_torch else np.mean(train_errors)
            if num_epochs>1: 
                # logger.info(f"\tColony({self.colony_id:3d}){thd}:: " + 
                #             f"Epoch: {(epoch):3d}/{num_epochs} " + 
                #             f"Epoch MSE: {mean_train_err}")
                if mean_train_err > 10:
                    break

        if self.use_torch:
            input = torch.tensor(data.test_input, requires_grad=False, dtype=torch.float32)
            target = torch.tensor(data.test_output, requires_grad=False, dtype=torch.float32)
        else:
            input = data.test_input
            target = data.test_output

        preds, errors, d_errors = self.single_thrust(input, target, prt=False, cal_gradient=False, cost_type=self.cost_type)

        test_errors   = torch.stack(errors) if self.use_torch else np.array(errors)
        # test_d_errors = torch.stack(d_errors) if self.use_torch else np.array(d_errors)
        # test_preds    = torch.stack(preds) if self.use_torch else np.array(preds)
        
        mean_train_err = torch.mean(train_errors) if self.use_torch else np.mean(train_errors)
        mean_test_err = torch.mean(test_errors) if self.use_torch else np.mean(test_errors)
        logger.info(f"Colony({self.colony_id}){thd}:: " + 
                    f"Training MSE: {mean_train_err:.6e}, " +
                    f"TEST MSE: {mean_test_err:.6e}")

        if self.use_torch:
            return torch.mean(test_errors).detach().numpy(), d_errors
        else:
            return np.mean(test_errors), d_errors



    def get_output(self):
        if self.use_torch:
            return torch.stack([node.out for node in self.out_nodes])
        else:
            return np.array([node.out for node in self.out_nodes])


    def plot_target_predict(self, data, file_name:str="")->None:
        if self.use_torch:
            input = torch.tensor(data.test_input, requires_grad=True, dtype=torch.float32)
            target = torch.tensor(data.test_output, requires_grad=True, dtype=torch.float32)
        else:
            input = data.test_input
            target = data.test_output   

        preds, errors, _ = self.single_thrust(input, target, prt=False, cal_gradient=True, cost_type=self.cost_type)
        errors   = torch.stack(errors) if self.use_torch else np.array(errors)
        preds    = torch.stack(preds) if self.use_torch else np.array(preds)
        
        
        E = torch.mean(errors).detach().numpy() if self.use_torch else np.mean(errors)
        logger.info(f"Colony({self.colony_id}):: Graph({self.id}): Testing Error {torch.mean(errors).item()}")   

        preds = preds.detach().numpy()
        target = target.detach().numpy()[self.lags:]
        array = np.column_stack((preds, target))
        
        np.savetxt(file_name+".txt", array, fmt="%f", header="Predicted,Target", comments="", delimiter=",")

        if file_name=="":
            file_name = f"colony_{self.colony_id}_struct_{self.id}_target_predict"
        fig, ax = plt.subplots()
        ax.plot(target, label='Target')
        ax.plot(preds, label='Predict')
        ax.legend()
        plt.savefig(file_name+".png")
        # plt.show()
        plt.cla()
        plt.clf()
        plt.close()

    def save_to_file(self, name):
        state = {}
        for k, v in self.__dict__.items():
            # handle torch tensors safely
            if hasattr(v, "detach") and hasattr(v, "cpu"):
                state[k] = v.detach().cpu().numpy()
            else:
                state[k] = v
        with open(name, "wb") as f:
            try:
                pickle.dump(state, f)
            except Exception as e:
                logger.error(f"Error saving structure to file: {e}")
                '''TODO: Fix the error below:: pickling error: RecursionError: maximum recursion depth exceeded'''

    @classmethod
    def load_from_file(cls, name):
        with open(name, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj

    def single_thrust_test(self, data):
        if self.use_torch:
            input = torch.tensor(data.test_input, requires_grad=False, dtype=torch.float32)
            target = torch.tensor(data.test_output, requires_grad=False, dtype=torch.float32)
        else:
            input = data.test_input
            target = data.test_output
        try:
            _,_,_ = self.single_thrust(input, target, prt=False, cal_gradient=True)
            return True
        except Exception as e:
            logger.error(f"Error in single_thrust_test: {e}")
            # raise e
            return False
        


    """ 
    ************************************ Visualization *************************************** 
    """
    def visualize_structure(self, filename=None,):
        dot = gv.Digraph(comment='Graph Visualization')
        dot.attr(rankdir='TB')
        
        # Add nodes
        in_nodes = []
        for node in self.nodes.values():
            if node.type == 0:
                # dot.node(str(node.id), label=f"N({node.id}) - p({node.point.get_id()})")
                if isinstance(node, GraphNode) and len(node.functions)==1:
                    dot.node(str(node.id), label=f"{function_names[list(node.functions.keys())[0]]}")
                else:
                    dot.node(str(node.id), label=f"N({node.id}) - p({node.point.get_id()})", shape='box', style='filled', fillcolor='lightgrey', rank='same')
            elif node.type==1:
                if node.point.name not in in_nodes:
                    in_nodes.append(node.point.name)
                    dot.node(str(node.point.name), label=f"N({node.id}) - p({node.point.get_id()}) {node.point.name}", shape='box', style='filled', fillcolor='lightblue', rank ='min', group='inputs', weight="0.5")
            elif node.type==2:
                dot.node(str(node.id), label=f"N({node.id}) - p({node.point.get_id()}) {node.point.name}", shape='box', style='filled', fillcolor='lightgrey', rank='max')
            else:
                print(f"Visualization: Unknown node type: Node({node.id} Type: {node.type})")
                exit(1)
        with dot.subgraph(name='cluster_same_rank') as s:
            s.attr(rank='min', style="invis")
            for node in self.in_nodes:
                s.node(node.point.name)
           

        
        
        # Add edges
        for node in self.nodes.values():
            for edge in node.outbound_edges.values():
                s = "solid"
                if  node.lag!=self.lags:
                    s = "dashed"
                if node.type==1:
                    dot.edge(str(node.point.name), str(edge.target.id), style=s, color='orange')
                else:
                    dot.edge(str(node.id), str(edge.target.id), style='solid', color='orange')

            continue
            for edge in node.inbound_edges.values():
                dot.edge(str(node.id), str(edge.source.id), style='dotted', color='red', arrowhead='none')
        
        # Render and save the structure
        if filename is None:
            filename = f"structure_{self.id}.gv"
        dot.render(filename, view=False)

    def generate_eqn(self, filename=None):
        eqn = ""
        for node in self.out_nodes:
            for edge in node.inbound_edges.values():
                eqn += f"{edge.source.get_eqn()} * {edge.get_weight():.4f} + "
            eqn = f"[{eqn[:-3]}] / {len(node.inbound_edges)}\n"
        if filename is not None:
            with open(filename, "w") as f:
                f.write(eqn)
        return eqn

    def generate_eqn_from_inputs(self, file_name=None,):
        eqn=""
        for node in self.out_nodes:
            eqn+= node.get_eqn_foactored([], self.lags)
        return eqn

    def write_structure(self, file_name=None):
        structure = f"Input Nodes ({len(self.in_nodes)}):\n"
        for node in self.in_nodes:
            structure+= f"\t{node.point.name}\n"
        
        num_inner_nodes=0
        num_edges = 0
        
        for n in self.nodes.values(): 
            if n.type==0: num_inner_nodes+=1
            num_edges+=len(n.outbound_edges)
        
        structure+= f"Number of Inner Nodes: {num_inner_nodes}\n"
        
        structure+= f"Output Nodes ({len(self.out_nodes)}):\n"
        for node in self.out_nodes:
            structure+=f"\t{node.point.name}\n"
        
        structure+=f"Number of Edges: {num_edges}\n"

        if file_name:
            with open(file_name, 'w') as f:
                f.write(structure)
        return structure


    def plot_path_points(self, ax=None, size=40, save_here=False, show=False, plt=None):
        if ax is None:
            fig = plt.figure(figsize=(size, size))
            ax = fig.add_subplot(111, projection='3d')
            save_here = True
            clear_here = True
        else:
            clear_here = False
        for i,points in enumerate(self.ants_paths):
            xs = [p.get_x() for p in points]
            ys = [p.get_y() for p in points]
            zs = [p.get_z() for p in points]
            fs = [p.get_f() for p in points]
            
            ax.scatter(xs, ys, zs, c=fs, cmap='viridis', marker='o', s=size)
            ax.plot(xs, ys, zs, color='black')#clrs[i])
            
            for i in range(len(xs) - 1):
                ax.quiver(xs[i], ys[i], zs[i], 
                xs[i+1] - xs[i], ys[i+1] - ys[i], zs[i+1] - zs[i], 
                arrow_length_ratio=0.1, color='b')
            for j in range(len(points)):
                ax.text(xs[j], ys[j], zs[j], points[j].get_id(), color='black', fontsize=10)
        if save_here:
            plt.savefig(f"structure_{self.id}_paths.png")
            if show:
                plt.show()
        if clear_here:
            plt.cla()
            plt.clf()
            plt.close('all')


    def plot_nodes(self, ax=None, size=400, save_here=False, show=False, plt=None):
        if ax is None:
            fig = plt.figure(figsize=(size, size))
            ax = fig.add_subplot(111, projection='3d')
            save_here = True
        for node in self.nodes.values():
            ax.scatter(node.point.get_x(), node.point.get_y(), node.z, color='red', marker='*', s=size)
            ax.text(node.point.get_x(), node.point.get_y(), node.z, f"{node.id}({node.point.get_id()})", color='red', fontsize=10)
            for edge in node.outbound_edges.values():
                ax.quiver(node.point.get_x(), node.point.get_y(), node.z, 
                            edge.target.point.get_x() - node.point.get_x(), 
                            edge.target.point.get_y() - node.point.get_y(), 
                            edge.target.z - node.z, 
                            arrow_length_ratio=0.0,
                            linewidth=10,
                            color='blue',
                            alpha=0.35,
                        )
            if node.type == 1:
                ax.text(node.point.get_x(), node.point.get_y(), node.z, f"{node.point.name}", color='black', fontsize=30)
        if save_here:
            plt.savefig(f"structure_{self.id}_nodes.png")
            if show:
                plt.show()
        
    def plot_pheromones(self, ax=None, size=40, save_here=False, show=False, plt=None):
        if ax is None:
            fig = plt.figure(figsize=(size, size))
            ax = fig.add_subplot(111, projection='3d')
            save_here = True

        # Distinct colors for each function index
        function_colors = {
            0: 'red',
            1: 'blue',
            2: 'green',
            3: 'orange',
            4: 'purple',
            5: 'brown',
            6: 'cyan',
            7: 'magenta',
            8: 'gold',
        }
        for point in self.space.points:
            func_index = round(point.get_f()*len(function_colors))  # assumes this returns an integer like 0-8
            color = function_colors.get(func_index, 'black')  # fallback color
            sc = ax.scatter(
                                point.get_x(), 
                                point.get_y(), 
                                point.get_z(), 
                                # c=point.get_f(), 
                                # cmap='viridis', 
                                color=color,
                                s=point.get_pheromone()*100, 
                                alpha=0.35, 
                                marker='o'
                            )           

    def plot_paths(self, ax=None, size=40, save_here=False, show=False, plt=None):
        if ax is None:
            fig = plt.figure(figsize=(size, size))
            ax = fig.add_subplot(111, projection='3d')
            save_here = True
        
        cmap = plt.get_cmap('tab10')  # Use a colormap with distinct colors
        colors = [cmap(i) for i in range(len(self.ants_paths))]

        ax.set_xlabel('Neural Width'    , fontsize=50, color='black', labelpad=10)
        ax.set_ylabel('Neural Depth'    , fontsize=50, color='black', labelpad=10)
        ax.set_zlabel('Time Lag'        , fontsize=50, color='black', labelpad=10)
        
        for i,points in enumerate(self.ants_paths):
            xs = [p.get_x() for p in points]
            ys = [p.get_y() for p in points]
            zs = [p.get_z() for p in points]
            fs = [p.get_f() for p in points]

            ax.text(xs[0], ys[0], zs[0], f"{points[0].name}", color=colors[i], fontsize=50)
            
            ax.scatter(xs, ys, zs, c=fs, cmap='viridis', marker='o', s=size)
            # ax.plot(xs, ys, zs, color=colors[i])
             
            # self.draw_cone(ax=ax, start=[xs[0], ys[0], zs[0]], direction=[xs[-1]-xs[0], ys[-1]-ys[0], zs[-1]-zs[0]], color=colors[i])


            for j in range(len(xs) - 1):
                ax.quiver(
                            xs[j], ys[j], zs[j], 
                            xs[j+1] - xs[j], ys[j+1] - ys[j], zs[j+1] - zs[j], 
                            arrow_length_ratio=0.0, 
                            linewidth=10, 
                            color=colors[i],
                            alpha=0.5
                        )
                
        if save_here:
            plt.savefig(f"structure_{self.id}_paths.png")
            if show:
                plt.show()
        if save_here:
            plt.cla()
            plt.clf()
            plt.close('all')
        # ipdb.set_trace()


    
