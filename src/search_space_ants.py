import sys
from typing import List
import numpy as np
import torch

DEFAULT_PHEROMONE_VALUE = 1.0
EVAPORATION_RATE = 0.9
MIN_PHEROMONE = 0.5
MAX_PHEROMONE = 10.0
NODE_TYPE = {"INPUT": 0, "HIDDEN": 1, "OUTPUT": 2}
DEFAULT_LAG = 10


class SearchSpaceANTS:
    def __init__(
        self,
        logger,
        inputs_names: List[str],
        outs_names: List[str],
        num_hid_nodes: int,
        num_hid_layers: int,
        lags: int = DEFAULT_LAG,
        max_pheromone: float = MAX_PHEROMONE,
        min_pheromone: float = MIN_PHEROMONE,
        evaporation_rate: float = EVAPORATION_RATE,
    ) -> None:
        self.logger = logger
        self.num_hid_nodes = num_hid_nodes
        self.num_hid_layers = num_hid_layers
        self.time_lags = lags
        self.all_points = {}
        self.evaporation_rate = evaporation_rate
        self.max_pheromone = max_pheromone
        self.min_pheromone = min_pheromone

        self.inputs_space = self.Inputs_Space(
            self.logger,
            inputs_names,
            lags,
            self.max_pheromone,
            self.min_pheromone,
            self.evaporation_rate,
        )
        self.output_space = self.Outputs_Space(outs_names)

        """Creating Hidden Points"""
        for lag in range(self.time_lags):
            for h in range(num_hid_layers):
                for n in range(num_hid_nodes):
                    pnt = self.Point(lag=lag, hid=h, point_type=NODE_TYPE["HIDDEN"])
                    self.all_points[pnt.id] = pnt

        """Linking Hidden Points with other Hidden Points and Output Points"""
        for in_pnt in self.all_points.values():
            for out_pnt in self.all_points.values():
                if (in_pnt.pos_l == out_pnt.pos_l and in_pnt.hid < out_pnt.hid) or (
                    in_pnt.pos_l < out_pnt.pos_l
                ):
                    link = self.Link(in_pnt, out_pnt)
                    in_pnt.fan_out[out_pnt] = link
                    out_pnt.fan_in.append(link)
            for out_pnt in self.output_space.points:
                link = self.Link(in_node=pnt, out_node=out_pnt)
                in_pnt.fan_out[out_pnt] = link
                out_pnt.fan_in.append(link)

        """Linking Input Points with Hidden Points and Output Points"""
        for input_level in self.inputs_space.inputs_spaces.values():
            for pnt in input_level.points:
                for hid_pnt in self.all_points.values():
                    if pnt.pos_l <= hid_pnt.pos_l:
                        link = self.Link(in_node=pnt, out_node=hid_pnt)
                        pnt.fan_out[hid_pnt] = link
                        hid_pnt.fan_in.append(link)
                for out_pnt in self.output_space.points:
                    link = self.Link(in_node=pnt, out_node=out_pnt)
                    pnt.fan_out[out_pnt] = link
                    out_pnt.fan_in.append(link)

    def evaporate_pheromone(
        self,
    ) -> None:
        self.inputs_space.evaporate_pheromone
        for pnt in self.all_points.values():
            for link in pnt.fan_out:
                link.pheromone = max(
                    link.pheromone * self.evaporation_rate, self.min_pheromone
                )

    class Point:
        counter = 0

        def __init__(
            self,
            lag: int,
            hid: int = None,
            point_type: int = None,
            name: str = None,
            inout_num: int = None,
        ) -> None:
            self.bias = torch.tensor(
                np.random.random(), dtype=torch.float32, requires_grad=True
            )
            self.fan_in = []
            self.fan_out = {}
            self.id = self.counter
            self.pos_l = lag  # time lag
            self.hid = hid
            self.inout_num = inout_num
            self.type = NODE_TYPE["HIDDEN"]
            self.pheromone = DEFAULT_PHEROMONE_VALUE
            self.name = name
            if self.name is not None:
                self.type = point_type
            SearchSpaceANTS.Point.counter += 1

    class Link:
        def __init__(self, in_node, out_node, weight: float = None) -> None:
            self.in_node = in_node
            self.out_node = out_node
            self.pheromone = DEFAULT_PHEROMONE_VALUE
            self.weight = weight
            if not self.weight:
                self.weight = np.random.random()

        def evaporate_pheromone(
            self,
        ) -> None:
            self.pheromone = max(self.pheromone * EVAPORATION_RATE, MIN_PHEROMONE)

        def increase_pheromone(self, pheromone_step) -> None:
            self.pheromone = min(self.pheromone + pheromone_step, MAX_PHEROMONE)

    class Single_Input_Space:
        def __init__(self, in_num: int, lags: int, name: str) -> None:
            self.time_lags = lags
            self.name: str = name
            self.in_num: int = in_num
            self.points: List[SearchSpaceANTS.Point] = []
            for lag in range(lags):
                self.points.append(
                    SearchSpaceANTS.Point(
                        lag=lag,
                        point_type=NODE_TYPE["INPUT"],
                        name=name,
                        inout_num=in_num,
                    )
                )
            self.input_pheromone = DEFAULT_PHEROMONE_VALUE

        def get_point(
            self,
        ):
            pheromones = [pnt.pheromone for pnt in self.points]
            norm_pheromones = pheromones / np.sum(pheromones)
            return np.random.choice(self.points, 1, p=norm_pheromones)[0]

    class Inputs_Space:
        def __init__(
            self,
            log,
            inputs_names,
            lags: int,
            max_pheromone,
            min_pheromone,
            evaporation_rate,
        ) -> None:
            self.inputs_spaces = {}
            self.inputs_names = inputs_names
            self.lags = lags
            self.max_pheromone = max_pheromone
            self.min_pheromone = min_pheromone
            self.evaporation_rate = evaporation_rate
            for i, name in enumerate(inputs_names):
                self.inputs_spaces[name] = SearchSpaceANTS.Single_Input_Space(
                    in_num=i, lags=lags, name=self.inputs_names[i]
                )
            if len(self.inputs_spaces) != len(inputs_names):
                log.error(f"Input Names Problem: {inputs_names}")
                sys.exit()

        def get_input(self, ant_exploration_rate: float):
            if np.random.random() < ant_exploration_rate:
                input_space = self.inputs_spaces[
                    self.inputs_names[
                        np.random.randint(low=0, high=len(self.inputs_names))
                    ]
                ]
            else:
                pheromones = [
                    input_space.input_pheromone
                    for input_space in self.inputs_spaces.values()
                ]
                norm_pheromones = pheromones / np.sum(pheromones)
                input_name = np.random.choice(self.inputs_names, 1, p=norm_pheromones)[
                    0
                ]
                input_space = self.inputs_spaces[input_name]

            if (
                np.random.random() < ant_exploration_rate
                or len(input_space.points) == 0
            ):
                return input_space.points[
                    np.random.randint(low=0, high=len(input_space.points))
                ]
            return input_space.get_point()

        def increase_pheromone(self, point, pheromone_step: float) -> None:
            new_phermone_value = (
                self.inputs_spaces[point.name].input_pheromone + pheromone_step
            )
            self.inputs_spaces[point.name].input_pheromone = min(
                new_phermone_value, self.max_pheromone
            )
            point.pheromone = min(point.pheromone + pheromone_step, self.max_pheromone)

        def evaporate_pheromone(
            self,
        ) -> None:
            for input_space in self.inputs_spaces.values():
                max(
                    input_space.input_pheromone * self.evaporation_rate,
                    self.min_pheromone,
                )
                points_to_be_removed = []
                for point in input_space.points:
                    max(point.pheromone * self.evaporation_rate, self.min_pheromone)
                    if point.pheromone < 0.05:
                        points_to_be_removed.append(point)
                for pnt in points_to_be_removed:
                    input_space.pop(pnt)

    class Outputs_Space:
        def __init__(self, outputs_names):
            self.points = [
                SearchSpaceANTS.Point(
                    lag=0,
                    hid=(i + 1) / len(outputs_names),
                    point_type=NODE_TYPE["OUTPUT"],
                    name=name,
                )
                for i, name in enumerate(outputs_names)
            ]
