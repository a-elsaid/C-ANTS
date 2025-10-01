"""
Ant class for the optimization agents
"""
import numpy as np
from search_space_cants import SearchSpaceCANTS as Space
from point import Point
from util import get_center_of_mass
import loguru

logger = loguru.logger
# from threading import Lock

logger = loguru.logger
logger.add("file_{time}.log")

# space_lock = Lock()

class Ant():
    def __init__(self, space: Space):
        self.x = 0
        self.y = 0
        self.z = 0
        self.f = 0
        self.sense_range = np.random.uniform(0.80, 0.90)
        self.explore_rate = np.random.uniform(0.2, 1.0)
        self.original_explore_rate = self.explore_rate
        self.mutation_sigma = 0.15
        self.space = space
        self.path = []
        self.best_behaviors = [] # [[RNN_performance, explore_rate, sense_range]]
        self.new_points = []
        self.new_in_points = [] 

    def update_position(self, point):
        self.x = point.get_x()
        self.y = point.get_y()
        self.z = point.get_z()
        self.f = point.get_f()
    
    def add_point_to_space(self, new_point):
        if new_point.get_node_type() == 0:
            self.new_points.append(new_point)
        elif new_point.get_node_type() == 1:
            self.new_in_points.append(new_point)
        else:  
            logger.error(f"Unexpeted Type For Creating New Point: {new_point.get_node_type()}")
            exit(1)
        self.update_position(new_point)

    def create_new_point(self, type):
        logger.debug(f"Creating new point (Type: {type})...")
        
        def create_coordinates(coord, lower_bound=0):
            if type==0:
                return np.random.uniform(
                                            max(0, coord - lower_bound), 
                                            min(coord + self.sense_range, 1)
                                        )
            elif type==1:
                return np.random.uniform(0, 1)
        # Create a new point based on the current position and pheromone levels
        lower_bound = self.sense_range
        x = create_coordinates(self.x, lower_bound=lower_bound)
        z = create_coordinates(self.z, lower_bound=lower_bound)
        f = create_coordinates(self.f, lower_bound=lower_bound)
        if type==0:
            if z<=self.z:
                lower_bound = 0
            y = create_coordinates(self.y, lower_bound=lower_bound)
            if y>=0.98:
                return self.pick_output()
        elif type==1:
             y = 0
        else:
            logger.error(f"Unexpeted Type For Creating New Point: {type}")
            exit(1)

        logger.trace(f"Creating Point: x={x}, y={y}, z={z}, f={f}, Type={type}")
        new_point =  Point( x=x, y=y, z=z, f=f, type=type)

        self.add_point_to_space(new_point)
        return new_point

    def opposite_to_pheromone_center(self, pheromone_center, type):
        if np.random.uniform() < self.explore_rate: # Explore Randomly
            return self.create_new_point(type)
        '''
        Create a new point opposite to the center of mass of pheromones
        to explit the search space
        '''
        logger.debug(f"Creating Point Opposite to Center of Mass (Type: {type})...")
        def create_coordinates(coord, cm):
            opposite_cm = coord - (cm - coord)
            opposite_cm = max(0, min(opposite_cm, 1))
            return opposite_cm

        lower_bound = self.sense_range
        x_ocfm = create_coordinates(self.x, pheromone_center[0])
        z_ocfm = create_coordinates(self.z, pheromone_center[2])
        f_ocfm = create_coordinates(self.f, pheromone_center[3])
        if type==0:
            y_ocfm = create_coordinates(self.y, pheromone_center[1])
            y_ocfm = max(y_ocfm, self.y)
            if y_ocfm>=0.98:
                return self.pick_output()
        elif type==1:
            y_ocfm = 0
        new_point = Point(x=x_ocfm, y=y_ocfm, z=z_ocfm, f=f_ocfm, type=type)
        self.add_point_to_space(new_point)
        logger.trace(f"Creating Point Opposite to Center of Mass: x={x_ocfm}, y={y_ocfm}, z={z_ocfm}, f={f_ocfm}, Type={type}")
        return new_point


    def pick_point(self, points, type):
        logger.debug(f"Picking Point (Type: {type})...")
        # Check if any of the output points are within the sense radius
        nearby_output_points = []
        if type == 0:   
            for o_point in self.space.output_points.values():
                if o_point.distance_to(x=self.x, y=self.y, z=self.z, f=self.f) <= self.sense_range:
                    nearby_output_points.append(o_point)
        
        if nearby_output_points:
            # If there are nearby output points, pick one of them
            return self.pick_output()

        # Calculate the center of mass of pheromones
        pheromone_points = []
        for point in points:
            if type == 0:
                if not(point.get_y() < self.y and point.get_z() < self.z):
                    if point.distance_to(x=self.x, y=self.y, z=self.z, f=self.f) <= self.sense_range:
                        pheromone_points.append(point)
            elif type == 1:
                if point.distance_to(x=self.x, y=self.y, z=self.z, f=self.f) <= self.sense_range:
                    pheromone_points.append(point)
            else:
                logger.error(f"Unexpeted Type For Picking Point: {type}")
                exit(1)
        pheromone_center = get_center_of_mass(pheromone_points)
        
        if pheromone_center is None:
            # If there are no pheromones, create a random new point
            return self.create_new_point(type)

        if np.random.uniform() < self.explore_rate:
            # Explore: pick a random point within the sense radius
            return self.opposite_to_pheromone_center(pheromone_center, type)
        else:
            # Exploit: choose the point with the highest pheromone level
            logger.debug(f"Creating Point at Center of Mass: {pheromone_center} (Type: {type}) ...")
            x, y, z, f = pheromone_center
            new_point = Point(x=x, y=y, z=z, f=f, type=type)
            self.add_point_to_space(new_point)
            return new_point
         
    def pick_input(self,):
        logger.debug("Picking INPUT")
        points_within_radius = []
        for point in self.space.input_points:
            if point.distance_to(x=self.x, y=self.y, z=self.z, f=self.f) <= self.sense_range:
                points_within_radius.append(point)
        chosen_point = self.pick_point(points_within_radius, type=1)
        if chosen_point.name == "":
            input_idx = round(chosen_point.get_x()*(len(self.space.input_names)-1))
            chosen_point.name = self.space.input_names[input_idx]
            chosen_point.name_idx = input_idx
        return chosen_point

    def pick_output(self,):
        logger.debug("Picking OUTPUT")
        def __pick_point():
            points = self.space.output_points.values()
            if np.random.uniform() < self.explore_rate:
                # Explore: pick a random point within the sense radius
                point = np.random.choice(list(points), size=1)[0]
                self.update_position(point)
                return point
            else:
                # Exploit: choose the point with the highest pheromone level
                expectation = np.random.uniform(0, max([p.get_pheromone() for p in points]))  # TODO: Adjust this value to be the max pheromone level
                for point in points:
                    if point.get_pheromone() > expectation:
                        self.update_position(point)
                        return point
                    expectation -= point.get_pheromone()
        return __pick_point()

    def choose_point(self,):
        points_within_radius = []
        for point in self.space.points:
            if (
                    not(point.get_x() < self.y and point.get_z() > self.z) and 
                    point.distance_to(x=self.x, y=self.y, z=self.z, f=self.f) <= self.sense_range
                ):
                points_within_radius.append(point)
        chosen_point = self.pick_point(points_within_radius, type=0)
        self.x, self.y, self.z, self.f = chosen_point.coordinates()
        return chosen_point

    def march(self,):
        self.path.append(self.pick_input())
        while not self.path[-1].is_output():
            self.path.append(self.choose_point())

    def reset(self,):
        self.x = 0
        self.y = 0
        self.z = 0
        self.f = 0
        self.path = []
        self.explore_rate = self.original_explore_rate
        self.new_points = []
        self.new_in_points = []


    """
    Setting up the Ant's Behavior using Genetic Evolution
    """
    def update_best_behaviors(self, fitness) -> None:
        """
        Updates the list of the ant's best behaviors based on the provided fitness value.

        If the list of best behaviors contains fewer than 10 entries, the current behavior
        (fitness, explore_rate, sense_range) is appended. If the list already contains 10 entries,
        the current behavior replaces the worst (last) entry only if its fitness is better (lower).
        After updating, the list is sorted in ascending order of fitness.

        Args:
            fitness (float): The fitness value of the current behavior to consider for inclusion.

        Returns:
            None
        """
        if len(self.best_behaviors) < 10:
            self.best_behaviors.append(
                [
                    fitness,
                    self.explore_rate,
                    self.sense_range,
                ]
            )
        else:
            if fitness < self.best_behaviors[-1][0]:
                self.best_behaviors[-1] = [
                    fitness,
                    self.explore_rate,
                    self.sense_range,
                ]
        self.best_behaviors.sort()

    def evolve_behavior(
        self,
    ) -> None:
        """
        using GA to evolve ant characteristics
        using cross over and mutations
        """

        def mutate():
            """
            perform mutations
            """
            (
                self.explore_rate,
                self.sense_range,
            ) = (
                np.random.uniform(low=0.1, high=0.9),
                np.random.uniform(low=0.1, high=0.9),
            )

        def cross_over(behavior1: np.ndarray, behavior2: np.ndarray):
            """
            perform cross over
            """
            # new_behavior = (behavior1 + behavior2) / 2
            new_behavior = list(
                ((np.subtract(behavior2[1:], behavior1[1:])) * np.random.random())
                + behavior1[1:]
            )
            (
                self.explore_rate,
                self.sense_range,
            ) = new_behavior

        if len(self.best_behaviors) < 10 or np.random.random() < self.mutation_sigma:
            # print("Mutating")
            mutate()
        else:
            # print("Crossing Over")
            indecies = np.arange(len(self.best_behaviors))
            indecies = np.random.choice(indecies, 2, replace=False)
            cross_over(
                self.best_behaviors[indecies[0]], self.best_behaviors[indecies[1]]
            )
