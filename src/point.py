import numpy as np
import loguru
import sys

NODE_TYPE = {"HIDDEN": 0, "INPUT": 1, "OUTPUT": 2}


logger = loguru.logger
# logger.add("file_{time}.log")
logger.add(sys.stdout, level="TRACE")

class Point:
    counter = 0

    def __init__(
        self,
        x, 
        y, 
        z, 
        type, 
        w = 0.0,
        f = 0.0,
        name="", 
        index=None,
    ) -> None:
        self.__x = x
        self.__y = y
        self.__z = z
        self.__w = w            # weight
        self.__f = f            # function type

        self.__pheromone = 1.0
        self.__node_type = type       # 0 = hid , 1 = input, 2 = output
        Point.counter += 1
        self.__id = Point.counter + 1
        self.name = name
        self.name_idx = index   # for input and output nodes
        logger.debug(f"Creating Point({self.__id}): x={x}, y={y}, z={z}, f={f}, Type={type}, Name={name}")


    def print_point(
        self,
    ) -> None:
        print(
            f"Pos X: {self.__x}, Pos Y: {self.__y}, Pos L: {self.__l}, " +
            f"Weight: {self.__w}, Function: {self.__f}, " +
            f"Type: {self.type}, Name: {self.name}, " +
            f"In_No: {self.inout_num}"
        )


    def get_pheromone(self):
        return self.__pheromone
    
    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def get_z(self):
        return self.__z
    
    def get_f(self):                              
        return self.__f

    def get_id(self):
        return self.__id

    def get_node_type(self):
        return self.__node_type
        
    def set_pheromone(self, p):
        self.__pheromone = p
    
    def set_z(self, z):
        self.__z = z
    
    def set_f(self, f):
        self.__f = f

    def set_id(self, id):
        self.__id = id


    def is_input(self):
        return self.__node_type == 1

    def is_output(self):
        return self.__node_type == 2
    
    def is_hidden(self):
        return self.__node_type == 0

    def coordinates(self):
        return self.__x, self.__y, self.__z, self.__f
    

    def distance_to(self, x, y, z, w=0.0, f=0.0):
        return np.sqrt(
                        (self.__x - x) ** 2 + 
                        (self.__y - y) ** 2 + 
                        (self.__z - z) ** 2 + 
                        (self.__w - w) ** 2 +
                        (self.__f - f) ** 2
                    )