import sys
from typing import List
import numpy as np
from point import Point
import loguru

logger = loguru.logger

class SearchSpaceCANTS:
    def __init__(
                self, 
                input_names: list, 
                output_names: list,
                evap_rate=0.1,
                lags=4,
    ) -> None:
        self.output_names = output_names
        self.input_names = input_names
        self.input_points = []
        self.output_points = {}
        self.points = []
        self.lag_levels = lags # Number of lag levels

        for idx, output_name in enumerate(self.output_names):
            # Output space is 1D x = [0, 1], y = 1, z = 1
            self.output_points[output_name] =  Point(
                                                x = idx / (len(input_names) - 1), 
                                                y = 1,          # y=0 and x between 0-1 and z=0
                                                z = 1,
                                                w =0.0,
                                                f = 0.0,
                                                type = 2,
                                                name = output_name,
                                                index = idx,
                                            )


    # def update_points(self, points, input_points, output_points):
    #     self.points = points
    #     self.input_points = input_points
    #     self.output_points = output_points

    # def get_all_points(self):
    #     return self.points,  self.input_points, self.output_points

    def add_new_points(self, new_points):
        self.points.extend(new_points)
    
    def add_input_points(self, input_points):
        self.input_points.extend(input_points)

    def evaporate_pheromone(self, evaporation_rate):
        logger.trace(f"Evaporating pheromone with rate: {evaporation_rate}")
        for point in self.points:
            point.set_pheromone(point.get_pheromone() * evaporation_rate)

        for point in self.input_points: # Evaporate output points pheromone
            pheromone = point.get_pheromone() * evaporation_rate
            
        for point in self.output_points.values(): # Evaporate output points pheromone
            pheromone = point.get_pheromone() * evaporation_rate
            if pheromone < 0.1:                  # Minimum pheromone level for output points
                point.set_pheromone(0.1)
            else:
                point.set_pheromone(pheromone)

        points_to_remove = [] # Points with low pheromone
        for point in self.points: # Remove points with low pheromone (inside nodes)
            if point.get_pheromone() < 0.1:
                points_to_remove.append(point)
        for point in points_to_remove:
            self.points.remove(point)
        
        points_to_remove = [] # Points with low pheromone
        for point in self.input_points: # Remove points with low pheromone (input nodes)
            if point.get_pheromone() < 0.1:
                points_to_remove.append(point)
        for point in points_to_remove:
            self.input_points.remove(point)


    def deposited_pheromone(self,graph):
        for node in graph.nodes.values():
            node.point.set_pheromone(node.point.get_pheromone() + 1)
            if node.point.get_pheromone() > 10:
                node.point.set_pheromone(10)
