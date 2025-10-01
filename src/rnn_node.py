import numpy as np
import torch
from loguru import logger
from helper import ACTIVATIONS
from point import Point
from edge import Edge, BNN_Edge

class Super_Node:
    """
    Super class for Graph_Node, RNN_Node, LSTM_Node, BNN_LSTM_Node
    """
    count = 0
    def __init__(
                    self, 
                    type=0, 
                    point: Point = None, 
                    lags=None,
                    use_torch=False,
    ):
        Super_Node.count += 1
        self.id = Super_Node.count
        self.use_torch = use_torch
        self.z = point.get_z()
        self.__cluster = None
        self.type = type                   # 0 = hidden, 1 = input, 2 = output
        self.functions = {}
        self.backfire = 0.0
        self.forefire = []
        self.d_node_value = None                  # Error value for output nodes only
        self.node_value = 0.0
        self.received_fire = 0
        self.received_backfire = 0
        self.point = point
        self.inbound_edges = {}         # incoming edges
        self.outbound_edges = {}        # outgoing edges
        self.fired = False
        self.bias = 0.0
        self.adjust_lag(lags=lags-1)    # Adjust lag levels based on the z value of the point

    def get_cluster(self):
        return self.__cluster

    def set_cluster(self, cluster):
        self.__cluster = cluster
    
    def adjust_lag(self, lags):
        self.lag = round(self.point.get_z() * lags)
        self.z = self.lag / max(1,lags)
        
    def compare_corr(self, node):
        return  (
                    self.point.get_x() == node.point.get_x() and 
                    self.point.get_y() == node.point.get_y() and 
                    self.point.get_z() == node.point.get_z()
        )

    def add_edge(self, to_node, weight=1.0):
        edge = Edge(self, to_node, use_torch=self.use_torch, weight=weight)
        if to_node.id == self.id:
            return
        logger.debug(f"Adding edge from {self.id} to {to_node.id}")
        self.outbound_edges[to_node.id] = edge
        to_node.inbound_edges[self.id]  = edge

    
    def add_fan_out_node(self, in_node, out_node, wght: float) -> None:
        if out_node not in self.outbound_edges:
            self.outbound_edges[out_node] = Edge(in_node=in_node, out_node=out_node, weight=wght)

    def add_fan_in_node(self, in_node) -> None:
        if in_node not in self.inbound_edges:
            self.inbound_edges[in_node] = Edge(in_node=in_node, out_node=self, weight=1.0)
            self.signals_to_receive += 1


class RNN_Node(Super_Node):
    """
    RNN node
    :param int id: node id
    :param Point point: search space point
    :param bias: node bias
    :param str activation_type: type of the activation function
    """

    def __init__(
        self,
        type=0,
        point: Point = None,
        lags: int | None = None,
        use_torch: bool = False,
        activation_type: str = "sigmoid",
    ) -> None:
        super().__init__(type, point, lags, use_torch)
        self.activation = ACTIVATIONS[activation_type]
        self.point = point
        self.type = self.point.get_node_type()  # 0 = normal, 1 = input, 2 = output
        """
        if self.type == 0:
            self.bias = torch.tensor(
                np.random.normal(), dtype=torch.float64, requires_grad=True
            )
        """
        

    def fire(
        self,
    ) -> None:
        """
        firing the node when activated
        """
        if self.use_torch:
            self.node_value = torch.stack(self.forefire).sum()
        else:
            self.node_value = np.sum(self.forefire)
        self.out = self.activation(self.node_value + self.bias)
        self.fired = True
        logger.trace(f"Node({self.id}) fired: {self.out}")
        for edge in self.outbound_edges.values():
            logger.debug(f"Edge ID({edge.id}) Node({edge.source.id})->Node({edge.target.id}) Weight({edge.weight})  ID:({id(edge.weight)})")
            edge.target.receive_fire(self.out * edge.weight)
        self.received_fire = 0
        self.forefire = []

    def receive_fire(self, signal: float) -> None:
        """
        receiving a synaptic signal and firing when all signals are received
        """
        self.received_fire+= 1
        self.fired = False
        logger.debug(f"Node({self.id}) received fire {signal}   [Signal({self.received_fire}/{len(self.inbound_edges)})]")
        self.forefire.append(signal)
        if self.received_fire >= len(self.inbound_edges):
            self.fire()

    def reset_node(
        self,
    ) -> None:
        with torch.no_grad():
            self.node_value = torch.tensor(0.0, dtype=torch.float64, requires_grad=False)
            self.out = torch.tensor(0.0, dtype=torch.float64, requires_grad=False)



class LSTM_Node(RNN_Node):
    def __init__(
         self,
        type=0,
        point: Point = None,
        lags: int | None = None,
        use_torch: bool = False,
        activation_type: str = "relu",
    ) -> None:
        super().__init__(type, point, lags, use_torch)
        self.wf, self.uf, self.wi, self.ui, self.wo, self.uo, self.wg, self.ug = self.gen_lstm_hid_state()
        self.ct = 0.0
        self.ht = 0.0

    def gen_lstm_hid_state(self):
        wf = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        uf = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        wi = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        ui = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        wo = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        uo = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        wg = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        ug = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        return wf, uf, wi, ui, wo, uo, wg, ug

    def fire(self) -> None:
        if self.use_torch:
            self.node_value = torch.stack(self.forefire).sum()
        else:
            self.node_value = np.sum(self.forefire)
        ft = self.activation(self.wf * self.node_value + self.uf * self.ht)
        it = self.activation(self.wi * self.node_value + self.ui * self.ht)
        ot = self.activation(self.wo * self.node_value + self.uo * self.ht)
        c_ = self.activation(self.wg * self.node_value + self.ug * self.ht)
        ct = ft * self.ct + it * c_
        self.ct = ct.detach()
        self.out = ot * self.activation(ct)
        self.ht = self.out.detach()
        self.fired = True
        logger.trace(f"Node({self.id}) fired: {self.out}")
        for edge in self.outbound_edges.values():
            logger.debug(f"Edge ID({edge.id}) Node({edge.source.id})->Node({edge.target.id}) Weight({edge.weight})  ID:({id(edge.weight)})")
            edge.target.receive_fire(self.out * edge.weight)
        
        self.recieved_fire = 0
        self.forefire = []

class BNN_LSTM_Node(LSTM_Node):
    def __init__(
         self,
        type=0,
        point: Point = None,
        lags: int | None = None,
        use_torch: bool = False,
        activation_type: str = "sigmoid",
        prior_mu=0,
        prior_sigma=5,
    ) -> None:
        super().__init__(type, point, lags, use_torch)
        self.ct = 0.0
        self.ht = 0.0
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        
        (
        self.wf_ro, 
        self.uf_ro, 
        self.wi_ro, 
        self.ui_ro, 
        self.wo_ro, 
        self.uo_ro, 
        self.wg_ro, 
        self.ug_ro ) = self.gen_lstm_hid_state()
        
        (
        self.wf_mu, 
        self.uf_mu, 
        self.wi_mu, 
        self.ui_mu, 
        self.wo_mu, 
        self.uo_mu, 
        self.wg_mu, 
        self.ug_mu ) = self.gen_lstm_hid_state()

        self.gates_ro = [self.wf_ro,
                         self.uf_ro,
                         self.wi_ro,
                         self.ui_ro,
                         self.wo_ro,
                         self.uo_ro,
                         self.wg_ro,
                         self.ug_ro]
        self.gates_mu = [self.wf_mu,
                         self.uf_mu,
                         self.wi_mu,
                         self.ui_mu,
                         self.wo_mu,
                         self.uo_mu,
                         self.wg_mu,
                         self.ug_mu]

        # self.wf, self.uf, self.wi, self.ui, self.wo, self.uo, self.wg, self.ug = self.gen_lstm_hid_state()


    def fire(self) -> None:
        self.waiting_signals = self.signals_to_receive

        ft_mu = self.activation(self.wf_mu * self.node_value + self.uf_mu * self.ht)
        ft_ro = self.activation(self.wf_ro * self.node_value + self.uf_ro * self.ht)
        ft_epsilon = torch.randn_like(ft_ro)
        ft_sample = ft_mu + torch.exp(ft_ro) * ft_epsilon
        ft = self.activation(ft_sample)

        it_mu = self.activation(self.wi_mu * self.node_value + self.ui_mu * self.ht)
        it_ro = self.activation(self.wi_ro * self.node_value + self.ui_ro * self.ht)
        it_epsilon = torch.randn_like(it_ro)
        it_sample = it_mu + torch.exp(it_ro) * it_epsilon
        it = self.activation(it_sample)

        ot_mu = self.activation(self.wo_mu * self.node_value + self.uo_mu * self.ht)
        ot_ro = self.activation(self.wo_ro * self.node_value + self.uo_ro * self.ht)
        ot_epsilon = torch.randn_like(ot_ro)
        ot_sample = ot_mu + torch.exp(ot_ro) * ot_epsilon
        ot = self.activation(ot_sample)

        c_mu = self.activation(self.wg_mu * self.node_value + self.ug_mu * self.ht)
        c_ro = self.activation(self.wg_ro * self.node_value + self.ug_ro * self.ht)
        c_epsilon = torch.randn_like(c_ro)
        c_sample = c_mu + torch.exp(c_ro) * c_epsilon
        c_ = self.activation(c_sample)

        ct = ft * self.ct + it * c_

        self.ct = ct.item()
        self.out = ot * self.activation(ct)
        self.ht = self.out.item()

        logger.trace(f"Node({self.id}) fired: {self.out}")
        
        for edge in self.outbound_edges.values():
            mu = self.out * edge.weight_mu
            ro = self.out * edge.weight_ro
            epsilon = torch.randn_like(ro)
            r = mu + torch.exp(ro) * epsilon

            edge.target.receive_fire(r)
            logger.trace(
                f"\t Node({self.id}) Sent Signal " +
                f"({r}) to Node({edge.target.id})"
            )
        self.fired = True




