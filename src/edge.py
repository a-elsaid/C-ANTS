import torch
import numpy as np

class Edge():
    count = 0
    def __init__(self, 
                 source, 
                 target, 
                 weight=1.0, 
                 use_torch=False
        ):
        self.id = Edge.count + 1
        Edge.count += 1
        self.source = source
        self.target = target
        self.grad = 0.0
        self.velocity = torch.tensor(0.0, dtype=torch.float64, requires_grad=True) if use_torch else 0.0

        if use_torch:
            self.weight = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
            # self.weight = torch.rand(1, dtype=torch.float64, requires_grad=True)
        elif weight is None:
            self.weight = np.random.uniform(-0.5, 0.5)
        else:
            self.weight = 1.0

    def get_weight(self):
        return self.weight.item() if isinstance(self.weight, torch.Tensor) else self.weight[0]


class BNN_Edge(Edge):
    def __init__(self, 
                 source, 
                 target,
                 prior_mu=0,
                 prior_sigma=5,
    ):
        super().__init__(source, target)
        self.source = source
        self.target = target
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.weight_mu = torch.tensor(
            float((np.random.normal()+self.prior_mu) * self.prior_sigma), 
            dtype=torch.float64, requires_grad=True
        )
        self.weight_ro = torch.tensor(
            float((np.random.normal()+self.prior_mu) * self.prior_sigma), 
            dtype=torch.float64, requires_grad=True
        )