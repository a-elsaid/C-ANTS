import numpy as np
from point import Point
from math import ceil, floor
from util import function_dict, function_names, d_function_dict
import loguru
import torch
import random
from edge import Edge

def sigmoid(x):
    x = max(-100.0, min(100.0, x))
    return 1 / (1 + np.exp(-x))

logger = loguru.logger


class GraphNode():
    count = 0
    def __init__(
                    self, 
                    type=0, 
                    point: Point = None, 
                    lags=None,
                    use_torch=False,
    ):
        GraphNode.count += 1
        self.id = GraphNode.count
        self.__use_torch = use_torch
        self.z = point.get_z()
        self.__cluster = None
        self.type = type                   # 0 = hidden, 1 = input, 2 = output
        self.backfire = 0.0
        self.forefire = []
        self.d_node_value = None                  # Error value for output nodes only
        self.out = 0.0
        self.received_fire = 0
        self.recieved_backfire = 0
        self.point = point
        self.inbound_edges = {}
        self.outbound_edges = {}
        
        self.functions = {}
        self.active = False
        if self.type == 2:
            self.functions = {0: function_dict[0]}
        else:
            self.pick_node_functions(function_dict)
        
        self.adjust_lag(lags=lags-1)    # Adjust lag levels based on the z value of the point


    def get_cluster(self):
        return self.__cluster

    def set_cluster(self, cluster):
        self.__cluster = cluster
        
    def compare_corr(self, node):
        return  (
                    self.point.get_x() == node.point.get_x() and 
                    self.point.get_y() == node.point.get_y() and 
                    self.point.get_z() == node.point.get_z()
        )

    def add_functions(self, functions):
        if self.type == 2:
            self.functions = {0: function_dict[0]}
        else:
            '''Add the functions to the node'''
            # self.functions.update(functions)

            '''Ramdomly pick a function and make it the only function for the node'''
            random_key = np.random.choice(list(functions.keys()))
            self.functions = {random_key: function_dict[random_key]}
    
    def pick_node_functions(self, function_dict):
        func_coord = self.point.get_f()
        func_id = (len(function_dict) - 1) * func_coord

        if int(func_id) == func_id:
            func_id = int(func_id)
            self.functions.update({func_id: function_dict[func_id]})
        elif int(func_id) == 0:
            self.functions.update({0: function_dict[0]})
        else:
            prev_func_id = int(func_id - 1)
            next_func_id = int(func_id + 1)
            # print(prev_func_id, next_func_id)   
            self.functions.update(
                                    {
                                        prev_func_id: function_dict[prev_func_id], 
                                        next_func_id: function_dict[next_func_id],
                                    }
            )
        random_id = random.choice(list(self.functions.keys()))
        self.functions = {random_id: self.functions[random_id]}


    def add_edge(self, to_node, weight=1.0):
        edge = Edge(self, to_node, use_torch=self.__use_torch, weight=weight)
        if to_node.id == self.id:
            return
        logger.debug(f"Adding edge from {self.id} to {to_node.id}")
        self.outbound_edges[to_node.id] = edge
        to_node.inbound_edges[self.id]  = edge


    def reset(self,):
        ... # Reset node state for a new forward pass

    def fire (self, ):
        results = []
        for fn_id, func in self.functions.items():
            if self.__use_torch:
                # fn_res = torch.clamp(func(torch.stack(self.forefire)), min=-1, max=1)
                fn_res = func(torch.stack(self.forefire))
            else:
                # fn_res = np.clip(func(self.forefire), -3.1, 3.1)
                fn_res = func(self.forefire)
            results.append(fn_res)
            self.out = torch.mean(torch.stack(results)) if self.__use_torch else np.mean(results)
            self.d_node_value = d_function_dict[fn_id](self.out) if self.__use_torch else d_function_dict[fn_id](self.out)
            
        logger.debug(f"Node({self.id:5d}) is firing {self.out:.5f}")
        for edge in self.outbound_edges.values():
            logger.debug(f"Edge ID({edge.id}) Node({edge.source.id})->Node({edge.target.id}) Weight({edge.weight})  ID:({id(edge.weight)})")
            node_value = self.out * edge.weight
            if self.__use_torch: node_value.retain_grad()
            edge.target.receive_fire(node_value)
        self.received_fire = 0
        self.forefire = []

    def receive_fire (self, signal):
        self.received_fire+= 1
        logger.debug(f"Node({self.id}) received fire {signal}   [Signal({self.received_fire}/{len(self.inbound_edges)})]")
        self.forefire.append(signal)
        if self.received_fire >= len(self.inbound_edges):
            self.fire()

    def update_weights(self, lr=0.001, momentum=0.1):
        def update_weights():
            for edge in self.inbound_edges.values():
                edge.source.update_weights()
                edge.velocity = momentum * edge.velocity + (edge.weight.grad if self.__use_torch else edge.grad)
                edge.weight -= torch.clamp(lr * edge.velocity, min=-10, max=10) if self.__use_torch else lr * edge.velocity
                edge.grad = 0.0
                if self.__use_torch:
                    edge.weight.grad.zero_()

        if self.__use_torch:
            with torch.no_grad():
                update_weights()
        else:
            update_weights()
            
            
    def fireback (self, err=None):
        if err is not None:
            self.backfire = err
            logger.debug(f"Node({self.id}) received backfire with error {err}")
        for edge in self.inbound_edges.values():
            edge.source.recieve_backfire(self.backfire * self.d_node_value * edge.weight)
            edge.grad+= self.backfire * edge.source.node_value * self.d_node_value
            edge.grad = np.clip(edge.grad, -0.5, 0.5)
        self.recieved_backfire = 0
        self.backfire = 0.0

    def recieve_backfire (self, value):
        self.recieved_backfire+= 1
        self.backfire+= value
        if self.recieved_backfire >= len(self.outbound_edges):
            self.fireback()

    def adjust_lag(self, lags):
        self.lag = round(self.point.get_z() * lags)
        self.z = self.lag / max(1,lags)


    def get_eqn(self):
        eqn = ""
        
        node_funs = list(self.functions.keys())
        operator = "+"
        if function_names[node_funs[0]] == "multiply":
            operator = "*"    

        if self.type == 1:
            return f"{self.point.name} "

        for edge in self.inbound_edges.values():
            eqn += f"{edge.source.get_eqn()} * {edge.weight} {operator} "
        
        eqn = f"({eqn[:-3]})"
        if function_names[node_funs[0]] not in ['multiply', 'add']:
            eqn = f"{node_funs[0]}{eqn}"

        return eqn

    def get_eqn_foactored(self, visited_nodes, max_lag):
        if self.id in visited_nodes or self.type==1: return ""
        visited_nodes.append(self.id)
        self_name = f"N{self.id}"
        if self.type==2: self_name = self.point.name
        eqn=f"{self_name} = {function_names[list(self.functions.keys())[0]]}("
        for edge in self.inbound_edges.values():
            if edge.source.type==1:
                node_name = edge.source.point.name + f"_{max_lag - self.lag}"
            else:
                node_name = f"N{edge.source.id}"
            eqn+=f"{edge.weight:.2f}*{node_name}, "
        eqn= eqn[:-2] + ")"
        if eqn!="": eqn+='\n'
        for edge in self.inbound_edges.values():
            eqn+= edge.source.get_eqn_foactored(visited_nodes, max_lag)
         
        return eqn
        
