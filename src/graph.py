from sklearn.cluster import DBSCAN
import numpy as np
from point import Point
from graph_node import GraphNode
from util import get_center_of_mass, function_names
import loguru
import graphviz as gv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from time import time
import sys
import torch
from structure import Structure

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

class Graph(Structure):
    def __init__(   self, 
                    ants_paths,
                    eps=0.25,
                    min_samples=2,
                    lags=5,
                    space=None,
                    colony_id=None,
                    use_torch=False,
                    cost_type="mse",
    ):
        super().__init__(ants_paths, space, GraphNode, lags, eps, min_samples, use_torch, colony_id, cost_type)
        
    

    def single_thrust(self, input, target, prt=False, cal_gradient=True, cost_type="mse", lr=0.001):
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
                
                errors.append(err)
                d_errors.append(d_err)
                for node, e in zip(self.out_nodes, d_err):
                    node.d_err = e
                if cal_gradient:
                    self.feed_backward(err)
            return preds, errors, d_errors

    
    def feed_forward(self, in_data):
        for i, node in enumerate(self.in_nodes):
            node.receive_fire(in_data[node.lag][node.point.name_idx])

    def feed_backward(self,err):
        if self.use_torch:
            torch.mean(err).backward()
        else:   
            for node in self.out_nodes:
                node.fireback(err)
        
        for node in self.out_nodes:
            node.update_weights(lr=0.01)
   

    def _dump_graph(self, fit, plot=False):
        
        self.visualize_graph(f"{self.out_dir}/colony_{self.id}_graph_{self.id}_fit_{fit}.gv")
        self.generate_eqn(f"{self.out_dir}/colony_{self.id}_graph_{self.id}_fit_{fit}.eqn")
        self.save_graph(self, f"{self.out_dir}/colony_{self.id}_graph_{self.id}_fit_{fit}.graph")
        if not plot:
            return
        self.write_structure(f"{self.out_dir}/colony_{self.id}_graph_{self.id}_fit_{fit}.strct")
        fig = plt.figure(figsize=(40, 40))
        ax = fig.add_subplot(111, projection='3d')
        # self.plot_path_points(ax=ax, plt=plt)
        self.plot_paths(ax=ax, plt=plt)
        # self.plot_nodes(ax=ax, plt=plt)
        self.plot_pheromones(ax=ax, plt=plt)

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

        # Manually create legend entries for whatever you added
        custom_legend_items = [
            Line2D([0], [0], marker='*', linestyle='None', markeredgecolor='red', markerfacecolor='red', markersize=80, label='Node'),
            Line2D([0], [0], linestyle='-', color='gray', label='Ant Path', linewidth=6)
            # Line2D([0], [0], marker='o', linestyle='None', markeredgecolor='gray', markerfacecolor='gray', markersize=80, label='Space Point'),
        ]

        for i, func_name in enumerate(function_names.values()):
            custom_legend_items.append(
                Line2D([0], [0], marker='o', linestyle='None', markeredgecolor=function_colors[i], markerfacecolor=function_colors[i], markersize=80, label=func_name)
            )

        # Add the custom legend
        ax.legend( 
                    handles=custom_legend_items,
                    loc='upper left',
                    fontsize=40,               # Increase font size
                    handlelength=4,            # Length of the legend handles
                    borderpad=1.0,             # Padding inside the legend box
                    labelspacing=1.2,          # Space between labels
                    handletextpad=1.5,         # Space between handle and text
                    frameon=True,              # Show legend frame
                    framealpha=1.0,            # Opaque box
                    borderaxespad=1.5          # Padding between legend and axes
        )
        plt.savefig(f"{self.out_dir}/colony_{self.id}_graph_{self.id}_fit_{fit}.png")
        plt.cla(); plt.clf(); plt.close()
        self.plot_target_predict(data=self.data, file_name=f"{self.out_dir}/colony_{self.id}_graph_{self.id}_fit_{fit}_target_predict", cost_type=self.cost_type)

        fig = plt.figure(figsize=(40, 40))
        ax = fig.add_subplot(111, projection='3d')
        self.plot_nodes(ax=ax, plt=plt)
        plt.savefig(f"{self.out_dir}/colony_{self.id}_nn_{self.id}_fit_{fit}.png")
        plt.cla(); plt.clf(); plt.close()
