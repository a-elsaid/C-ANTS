""" RNN class """
import sys
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import torch
from loguru import logger
from helper import ACTIVATIONS, LOSS
from rnn_node import *
from structure import Structure
from search_space_cants import SearchSpaceCANTS
from search_space_ants import SearchSpaceANTS


class RNN(Structure):
    def __init__(
            self,
            ants_paths,
            space: SearchSpaceCANTS | SearchSpaceANTS,
            eps: float = 0.25,
            lags: int = 5,
            min_samples: int = 2,
            use_torch: bool = True,
            loss_fun: str = "mse",
            act_fun: str = "sigmoid",
            colony_id: int = None,
            worker: int = None,
            cost_type: str = "mse",
        ) -> None:
        super().__init__(ants_paths, space, LSTM_Node, lags, eps, min_samples, use_torch, colony_id, cost_type)
        self.fitness: float = None
        self.uncertainity = 0.
        self.mean_bnn_fit = 0.
        self.score = 0.
        self.err: torch.float64 = torch.tensor(0.0, dtype=torch.float64)
        self.total_err = 0.0
        self.act_fun = act_fun
        self.loss_fun = LOSS[loss_fun]
        self.lags = lags
        self.colony_id = colony_id
        self.worker = worker


    def build_fully_connected_rnn(
        self, input_names, output_names, lags, hid_layers, hid_nodes
    ) -> None:
        from search_space_ants import RNNSearchSpaceANTS

        self.lags = lags
        Point = RNNSearchSpaceANTS.Point
        for i, name in enumerate(input_names):
            for lag in range(lags):
                node = LSTM_Node(Point(None, None, 0, name, i), lag, self.act_fun)
                self.nodes[node.id] = node
                self.in_nodes.append(node)

        for name in output_names:
            node = LSTM_Node(Point(None, None, 2, name, i), lags - 1, self.act_fun)
            self.nodes[node.id] = node
            self.out_nodes.append(node)

        next_layer = self.out_nodes
        for _ in range(hid_layers):
            curr_layer = []
            for _ in range(hid_nodes):
                for lag in range(lags):
                    node = LSTM_Node(Point(None, None, 1, None, None), lag, self.act_fun)
                    self.nodes[node.id] = node
                    curr_layer.append(node)
            for in_node in curr_layer:
                for out_node in next_layer:
                    in_node.add_fan_out_node(in_node, out_node, np.random.normal())
                    out_node.add_fan_in_node(in_node)
            next_layer = curr_layer

        for in_node in self.in_nodes:
            for out_node in next_layer:
                in_node.add_fan_out_node(in_node, out_node, np.random.normal())
                out_node.add_fan_in_node(in_node)

        for node in self.nodes.values():
            node.waiting_signals = node.signals_to_receive
            logger.trace(
                f"Node({node.id}): Type: {node.type}  Name: {node.point.name} " +
                f"Point: {node.point.get_id()}"
            )
            logger.trace(
                f"\tOut Nodes: {[edge.target.id for edge in node.outbound_edges.values()]}"
            )
            logger.trace(f"\tIn Nodes: {[node.id for node in node.inbound_edges.values()]}")


    def generate_bnn_version(self,) -> None:

        self.bnn_nodes = {}
        self.bnn_input_nodes = []
        self.bnn_output_nodes = []
        for n_id, node in self.nodes.items():
            if type(node)==LSTM_Node:
                bnn_node = BNN_LSTM_Node(node.point, node.lag, activation_type='sigmoid')
                bnn_node.id = node.id
                self.bnn_nodes[node.id] = bnn_node
            else:
                raise("BNN for None LSTM is not implemented yet")

            if node in self.in_nodes:
                self.bnn_input_nodes.append(self.bnn_nodes[node.id])
            elif node in self.out_nodes:
                self.bnn_output_nodes.append(self.bnn_nodes[node.id])

        for node in self.nodes.values():
            for rnn_edge in node.outbound_edges.values():
                in_node  = self.bnn_nodes[rnn_edge.source.id]
                out_node = self.bnn_nodes[rnn_edge.target.id]
                # edge = BNN_Edge(e_id=rnn_edge.id, in_node=in_node, out_node=out_node)
                # in_node.outbound_edges[out_node] = edge

                out_node.add_fan_in_node(in_node)
                in_node.add_fan_out_node(e_id=rnn_edge.id,  out_node=out_node)

        for node in self.bnn_nodes.values():
            node.waiting_signals = node.signals_to_receive
            logger.trace(
                f"Node({node.id}) Point({node.point.get_id()}) Type: {node.type} " +
                f"WaitingSignals: {node.waiting_signals}"
            )
                
    def feed_forward(
        self,
        inputs: np.ndarray,
        active_inference: bool = False,
    ) -> None:
        """
        feeding forward a data point to get an output
        """

        """
        for node in self.nodes.values():
            node.waiting_signals = node.signals_to_receive
            node.fired = False
        """
        if active_inference:
            nodes = self.bnn_nodes
            input_nodes  = self.bnn_input_nodes
            output_nodes = self.bnn_output_nodes
        else:
            nodes = self.nodes
            input_nodes  = self.in_nodes
            output_nodes = self.out_nodes

        for node in input_nodes:
            node.receive_fire(inputs[node.lag][node.point.name_idx])

        raise_err = False
        for n in nodes.values():
            if not n.fired:
                logger.error(f"Node {n.id} (type: {n.type}) Didn't Fire  Point: {n.point.get_id()}")
                raise_err = True
        if raise_err:
            raise RuntimeError("Stopping: One or more nodes DID NOT Fire")

    def bnn_k_loss(self,):
        kl = 0
        for n in self.bnn_nodes.values():
            for ro,mu in zip(n.gates_mu, n.gates_ro):
                kl+=(ro.item()-np.log(n.prior_sigma)) + \
                    (ro.item()**2 + (mu.item() - n.prior_mu)**2) / (2*n.prior_sigma**2) - 0.5

            for e in n.outbound_edges.values():
                kl+=(e.weight_ro.item()-np.log(n.prior_sigma)) + \
                    (e.weight_ro.item()**2 + (e.weight_mu.item() - n.prior_mu)**2) / (2*n.prior_sigma**2) - 0.5
        
        return kl

    def single_thrust(
        self, 
        input: np.ndarray, 
        target: np.ndarray, 
        cal_gradient=True,
        prt=False,
        cost_type="mse",
        active_inference=False,
        bnn_kl_param=.01,
        lr=0.001,
    ) -> None:
        """
        perform one epoch using the whole dataset
        """
        if active_inference:
            nodes = self.bnn_nodes
        else:
            nodes = self.nodes

        preds, errors, d_errors = [], [], []
        for i in range(0, len(input) - self.lags):
            
            input_data, target_data = input[i:i+max(1,self.lags)], target[i+self.lags]
            self.feed_forward(input_data)

            out = self.get_output()

            if cost_type == "bicross_entropy":
                if isinstance(out, torch.Tensor): # torch tensor
                    out = torch.sigmoid(out)      # Ensure out is in the range [0, 1] for bicross entropy
                else:
                    out = sigmoid(out)
                err, d_err = self.bicross_entropy(target_data, out, prt)
            elif cost_type == "cross_entropy":
                if isinstance(out, torch.Tensor): # torch tensor
                    out = torch.softmax(out, dim=0)  # Apply softmax to ensure probabilities sum to
                else:
                    out = softmax(out) # Apply softmax to ensure probabilities sum to 1
                err, d_err = self.cross_entropy(target_data, out, prt)
            elif cost_type == "mse":  # default is mse
                if isinstance(out, torch.Tensor): # torch tensor
                    out = torch.sigmoid(out) # apply sigmoid to ensure output is in the range [0, 1]
                else:
                    out = sigmoid(out) # apply sigmoid to ensure output is in the range [0, 1]
                
                err, d_err = self.mse(target_data, out, prt)
            else:
                raise ValueError(f"Unknown error type: {type}. Supported types are 'mse', 'bicross_entropy', and 'cross_entropy'.")
            preds.append(out)
            
            kl = 0.0
            if active_inference:
                kl = self.bnn_k_loss()
            err = err + bnn_kl_param * kl

            errors.append(err)
            d_errors.append(d_err)
            for node, e in zip(self.out_nodes, d_err):
                node.d_err = e
            if cal_gradient:
                self.feed_backward(err, lr=lr)
            
            return preds, errors, d_errors

    def feed_backward(self, err, lr, active_inference=False) -> None:
        """
        calculating gradients
        """

        # torch.autograd.set_detect_anomaly(True)

        if active_inference:
            nodes = self.bnn_nodes
        else:
            nodes = self.nodes
        logger.debug(f"ERR: {err}")
        err.backward()
        with torch.no_grad():
            for node in nodes.values():
                for edge in node.outbound_edges.values():
                    if active_inference:
                        logger.trace(
                            f"From Point {node.point.get_id()} To Point \
                                {edge.target.point.get_id()}: \
                                dweight_mu = {edge.weight_mu.grad} \
                                dweight_ro = {edge.weight_ro.grad}"
                        )
                        logger.trace(f"\t Weight before update: dw_mu: {edge.weight_mu}, dw_ro: {edge.weight_ro}")
                        edge.weight_mu += edge.weight_mu.grad * lr
                        edge.weight_ro += edge.weight_ro.grad * lr
                        logger.trace(f"\t Weight after update: dw_mu: {edge.weight_mu}, dw_ro: {edge.weight_ro}")
                        edge.weight_mu.grad.zero_()
                        edge.weight_ro.grad.zero_()
                    else:
                        logger.trace(
                            f"From Node({node.id}) (type: {node.type}) ➡︎ " +
                            f"Node{edge.target.id} (type: {edge.target.type}):: dweight = {edge.weight.grad}"
                        )
                        logger.trace(f"\t Weight before update: {edge.weight}")
                        edge.weight += edge.weight.grad * lr
                        logger.trace(f"\t Weight after update: {edge.weight}")
                        edge.weight.grad.zero_()

                if active_inference:
                    if isinstance(node, BNN_LSTM_Node):
                        node.wf_mu -= node.wf_mu.grad * lr
                        node.uf_mu -= node.uf_mu.grad * lr
                        node.wi_mu -= node.wi_mu.grad * lr
                        node.ui_mu -= node.ui_mu.grad * lr
                        node.wo_mu -= node.wo_mu.grad * lr
                        node.uo_mu -= node.uo_mu.grad * lr
                        node.wg_mu -= node.wg_mu.grad * lr
                        node.ug_mu -= node.ug_mu.grad * lr

                        node.wf_ro -= node.wf_ro.grad * lr
                        node.uf_ro -= node.uf_ro.grad * lr
                        node.wi_ro -= node.wi_ro.grad * lr
                        node.ui_ro -= node.ui_ro.grad * lr
                        node.wo_ro -= node.wo_ro.grad * lr
                        node.uo_ro -= node.uo_ro.grad * lr
                        node.wg_ro -= node.wg_ro.grad * lr
                        node.ug_ro -= node.ug_ro.grad * lr
                        if torch.isnan(node.wf_ro.grad):
                            exit()
                        node.wf_mu.grad.zero_()
                        node.uf_mu.grad.zero_()
                        node.wi_mu.grad.zero_()
                        node.ui_mu.grad.zero_()
                        node.wo_mu.grad.zero_()
                        node.uo_mu.grad.zero_()
                        node.wg_mu.grad.zero_()
                        node.ug_mu.grad.zero_()

                        node.wf_ro.grad.zero_()
                        node.uf_ro.grad.zero_()
                        node.wi_ro.grad.zero_()
                        node.ui_ro.grad.zero_()
                        node.wo_ro.grad.zero_()
                        node.uo_ro.grad.zero_()
                        node.wg_ro.grad.zero_()
                        node.ug_ro.grad.zero_()
                else:
                    if isinstance(node, LSTM_Node):
                        # print(f"Updating LSTM Node {node.id} type: {node.type} -- {type(node)}", node.wf.grad, lr)
                        node.wf -= node.wf.grad * lr
                        node.uf -= node.uf.grad * lr
                        node.wi -= node.wi.grad * lr
                        node.ui -= node.ui.grad * lr
                        node.wo -= node.wo.grad * lr
                        node.uo -= node.uo.grad * lr
                        node.wg -= node.wg.grad * lr
                        node.ug -= node.ug.grad * lr
                        if torch.isnan(node.wf.grad):
                            exit()
                        node.wf.grad.zero_()
                        node.uf.grad.zero_()
                        node.wi.grad.zero_()
                        node.ui.grad.zero_()
                        node.wo.grad.zero_()
                        node.uo.grad.zero_()
                        node.wg.grad.zero_()
                        node.ug.grad.zero_()

    def copy_rnn(self,) -> List:
        nodes = self.nodes
        rnn_info = []
        rnn_info.append(self.id)
        rnn_info.append(self.fitness)
        rnn_info.append(self.score)
        with torch.no_grad():
            for node in nodes.values():
                for edge in node.outbound_edges.values():
                    rnn_info.append(edge.weight.detach().numpy())

                if isinstance(node, LSTM_Node):
                    rnn_info.append(node.wf.detach().numpy())
                    rnn_info.append(node.uf.detach().numpy())
                    rnn_info.append(node.wi.detach().numpy())
                    rnn_info.append(node.ui.detach().numpy())
                    rnn_info.append(node.wo.detach().numpy())
                    rnn_info.append(node.uo.detach().numpy())
                    rnn_info.append(node.wg.detach().numpy())
                    rnn_info.append(node.ug.detach().numpy())
        return rnn_info

    def assign_rnn(self, rnn_info) -> List:
        nodes = self.nodes
        self.fitness = rnn_info[1]
        self.score = rnn_info[2]
        i = 3
        with torch.no_grad():
            for node in nodes.values():
                for edge in node.outbound_edges.values():
                    edge.weight.detach_()
                    edge.weight = torch.from_numpy(rnn_info[i])
                    edge.weight.requires_grad_(True)
                    i+=1

                if isinstance(node, LSTM_Node):
                    node.wf.detach_()
                    node.wf = torch.from_numpy(rnn_info[i])
                    node.wf.requires_grad_(True) 
                    i+=1
                    node.uf.detach_()
                    node.uf = torch.from_numpy(rnn_info[i])
                    node.uf.requires_grad_(True) 
                    i+=1
                    node.wi.detach_()
                    node.wi = torch.from_numpy(rnn_info[i])
                    node.wi.requires_grad_(True) 
                    i+=1
                    node.ui.detach_()
                    node.ui = torch.from_numpy(rnn_info[i])
                    node.ui.requires_grad_(True) 
                    i+=1
                    node.wo.detach_()
                    node.wo = torch.from_numpy(rnn_info[i])
                    node.wo.requires_grad_(True) 
                    i+=1
                    node.uo.detach_()
                    node.uo = torch.from_numpy(rnn_info[i])
                    node.uo.requires_grad_(True) 
                    i+=1
                    node.wg.detach_()
                    node.wg = torch.from_numpy(rnn_info[i])
                    node.wg.requires_grad_(True) 
                    i+=1
                    node.ug.detach_()
                    node.ug = torch.from_numpy(rnn_info[i])
                    node.ug.requires_grad_(True) 
                    i+=1

    def test_rnn(self, 
                 inputs: np.ndarray, 
                 outputs: np.ndarray, 
                 loss_fun=None, 
                 active_inference=False,
                 bnn_kl_param=.01,
        ) -> None:
        """
        feeding forward a all testing data points and getting
        the output to calculate the error between the output
        and the groundtruth
        """

        with torch.no_grad():
            if not loss_fun:
                loss_fun = self.loss_fun
            err = 0.0
            for i in range(len(inputs) - self.lags):
                res = self.feed_forward(inputs[i : i + self.lags], active_inference=active_inference)


                kl=0.0
                if active_inference:
                    kl = self.bnn_k_loss()

                e = [loss+bnn_kl_param*kl for loss in loss_fun(res, outputs[i])]
                e = sum(e) / len(e)
                err += e

            if not active_inference:
                self.fitness = err.item() / i
        return err.item()/i
