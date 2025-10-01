import numpy as np

def get_center_of_mass(points):
        center_pheromone_x = 0.0
        center_pheromone_y = 0.0
        center_pheromone_z = 0.0
        center_pheromone_f = 0.0
        total_pheromone = 0.0
        for point in points:
            pheromone = point.get_pheromone()
            center_pheromone_x += point.get_x() * pheromone
            center_pheromone_y += point.get_y() * pheromone
            center_pheromone_z += point.get_z() * pheromone
            center_pheromone_f += point.get_f() * pheromone
            total_pheromone += pheromone
        if total_pheromone == 0:
            return None
        return [
            center_pheromone_x / total_pheromone,
            center_pheromone_y / total_pheromone,
            center_pheromone_z / total_pheromone,
            center_pheromone_f / total_pheromone,
        ]


def dbscan(points, epsilon, min_pts):
    def distance(p1, p2):
        d = np.sqrt(
                    np.power(
                        np.sum(
                                np.subtract(p1, p2),
                                2
                        )
                    )
            )

        return d
        
    def expand_cluster(cluster, point, neighbors, points, epsilon, min_pts, visited):
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                new_neighbors = get_neighbors(neighbor, points, epsilon)
                
                if len(new_neighbors) >= min_pts:
                    neighbors.extend(new_neighbors)
            
            if neighbor not in [p for p in cluster]:
                cluster.append(neighbor)


    def get_neighbors(point, points, epsilon):
        neighbors = []
        for p in points:
            if distance(point, p) <= epsilon:
                neighbors.append(p)
        return neighbors

    clusters = []
    visited = set()
    
    for point in points:
        if point in visited:
            continue
        visited.add(point)
        
        neighbors = get_neighbors(point, points, epsilon)
        
        if len(neighbors) < min_pts:
            continue
        
        cluster = [point]
        expand_cluster(cluster, point, neighbors, points, epsilon, min_pts, visited)
        clusters.append(cluster)
    
    return clusters

import torch
# Functions


def relu(x):
    def relu_fun():
        return max(0, x)
    if type(x) == torch.Tensor:
        return torch.mean(torch.clamp(x, min=0))
    return np.mean(list(torch.max(0, x)))

def leaky_relu(x):
    def leaky_relu_fun():
        return max(0.01*x, x)
    if type(x) == torch.Tensor:
        return torch.mean(torch.max(0.01*x, x))
    return np.mean(list(map(leaky_relu_fun, x)))

def tanh(x):
    if type(x) == torch.Tensor:
        return torch.mean(torch.tanh(x))
    return np.tanh(np.mean(x))

def negate(x):
    if type(x) == torch.Tensor:
        return torch.mean(x * -1)
    return np.mean(np.array(x) * -1)

def maximum(x):
    if type(x) == torch.Tensor:
        return torch.max(x)
    return np.max(x)

def minimum(x):
    if type(x) == torch.Tensor:
        return torch.min(x) 
    return np.min(x)

def logic_and(x):
    if type(x) == torch.Tensor:
        return torch.sum(x > 0)
    return np.sum(np.array(x) > 0)

def logic_or(x):
    if type(x) == torch.Tensor:
        return torch.sum(x > 0)
    return np.sum(np.array(x) > 0) > 0

def exp(x):
    if type(x) == torch.Tensor:
        return torch.mean(torch.exp(x))
    return np.mean(np.exp(x))

def log(x, base=np.e):
    if type(x) == torch.Tensor:
        y = torch.clip(x, 1e-10, None)
        if base == np.e:
            return torch.mean(torch.log(y))
        else:
            return torch.mean(torch.log(y) / np.log(base))
    x = np.clip(x, 1e-10, None)
    if base == np.e:
        return np.mean(np.log(x))  # Natural log
    else:
        return np.mean(np.log(x) / np.log(base))  # Log with custom base

def sqrt(x):
    if type(x) == torch.Tensor:
        signs = torch.sign(x)
        return torch.mean(signs*torch.sqrt(torch.abs(x)))
    signs = np.sign(x)
    return np.mean(signs*np.sqrt(np.abs(x)))

def square(x):
    if type(x) == torch.Tensor:
        return torch.mean(torch.square(torch.abs(x)))
    return np.mean(np.square(np.abs(x)))

def cube(x):
    if type(x) == torch.Tensor:
        return torch.mean(torch.pow(x, 3))
    return np.mean(np.power(x, 3))

def softmax(x):
    if type(x) == torch.Tensor:
        return torch.argmax(torch.exp(x)) / torch.sum(torch.exp(x))/len(x)
    return np.argmax(np.exp(x)) / np.sum(np.exp(x))/len(x)

def cosh(x):
    if type(x) == torch.Tensor:
        return torch.mean(torch.cosh(x))
    return np.mean(np.cosh(x))

def sinh(x):
    if type(x) == torch.Tensor:
        return torch.mean(torch.sinh(x))
    return np.mean(np.sinh(x))

def xor(x):
    # XOR over multiple inputs is true if an odd number of inputs are true
    if type(x) == torch.Tensor:
        return torch.mean(x) % 2
    return np.mean(x) % 2  # Returns 1 if the count of 1s is odd, 0 if even

def xnor(x):
    # XNOR is the complement of XOR, true if an even number of inputs are true
    return 1 - xor(x)  # Returns 1 if XOR result is 0, 0 if XOR result is 1

def ispositive(x):
    if type(x) == torch.Tensor:
        return torch.sum(x > 0) > 0
    return np.sum(np.array(x) > 0) > 0

def isnegative(x):
    if type(x) == torch.Tensor:
        return torch.sum(x < 0) < 0
    return np.sum(np.array(x) < 0) < 0

def iszero(x):
    if type(x) == torch.Tensor:
        s = torch.sum(x)
        return s>-0.05 and s<0.05
    s = np.sum(x)
    return s>-0.05 and s<0.05

'''    
function_dict = {
                    0: add, 
                    1: multiply, 
                    2: sin, 
                    3: cos,
                    4: tan,
                    5: sigmoid,
                    6: relu,
                    7: leaky_relu,
                    8: tanh,
                    9: inverse,
                    10: negate,
                    11: maximum,
                    12: minimum,
                    13: logic_and,
                    14: logic_or,
                    15: exp,
                    16: log,
                    17: sqrt,
                    18: square,
                    19: cube,
                    20: softmax,
                    21: cosh,
                    22: sinh,
                    23: xor,
                    24: xnor,
                    25: ispositive,
                    26: isnegative,
                    27: iszero,
                    28: mean,
                }
function_names = {
                    0: 'add', 
                    1: 'multiply', 
                    2: 'sin', 
                    3: 'cos',
                    4: 'tan',
                    5: 'sigmoid',
                    6: 'relu',
                    7: 'leaky_relu',
                    8: 'tanh',
                    9: 'inverse',
                    10: 'negate',
                    11: 'maximum',
                    12: 'minimum',
                    13: 'logic_and',
                    14: 'logic_or',
                    15: 'exp',
                    16: 'log',
                    17: 'sqrt',
                    18: 'square',
                    19: 'cube',
                    20: 'softmax',
                    21: 'cosh',
                    22: 'sinh',
                    23: 'xor',
                    24: 'xnor',
                    25: 'ispositive',
                    26: 'isnegative',
                    27: 'iszero',
                    28: 'mean',
                }
'''



def add(x):
    if type(x) == torch.Tensor:
        return torch.sum(x)
    return np.sum(x)

def multiply(x):
    if type(x) == torch.Tensor:
        return torch.prod(x)
    return np.prod(x)

def sin(x):
    if type(x) == torch.Tensor:
        return torch.sin(torch.sum(x))
    return np.mean(list(map(np.sin, x)))

def cos(x):
    if type(x) == torch.Tensor:
        return torch.cos(torch.sum(x))
    return np.mean(list(map(np.cos, x)))

def tan(x):
    if type(x) == torch.Tensor:
        return torch.tan(torch.sum(x))
    return np.mean(list(map(np.tan, x)))

def sigmoid(x):
    if type(x) == torch.Tensor:
        return torch.sigmoid(torch.sum(x))
    def sig_fun(x):
        x = np.float64(x)
        return 1 / (1e-9 + 1 + np.exp(-x))
    return np.mean(list(map(sig_fun, x)))

def inverse(x):
    if type(x) == torch.Tensor:
        return torch.mean(1/(x+1e-8))
    x = np.array(x)
    return np.mean(1/(x+1e-8))

def mean(x):
    if type(x) == torch.Tensor:
        return torch.mean(x)
    x = np.array(x)
    return np.mean(x)


def d_add(x):
    if type(x) == torch.Tensor:
        return torch.tensor(1.0, dtype=torch.float64)
    return 1.0

def d_multiply(x):
    if type(x) == torch.Tensor:
        return torch.tensor(1.0, dtype=torch.float64)
    return 1.0

def d_sin(x):
    if type(x) == torch.Tensor:
        return torch.cos(x)
    return np.cos(x)

def d_cos(x):
    if type(x) == torch.Tensor:
        return torch.sin(x) * -1
    return np.cos(x) * -1

def d_tan(x):
    if type(x) == torch.Tensor:
        return 1/(torch.cos(x))**2
    return 1/((np.cos(x))**2+1e-9)

def d_tanh(x):
    if type(x) == torch.Tensor:
        return 1 - (torch.tanh(x))**2
    return 1 - (np.tanh(x))**2

def d_sigmoid(x):
    return x/(1-x+1e-9)

def d_inverse(x):
    return -1/(x+1e-9)**2

def d_mean(x):
    if type(x) == torch.Tensor:
        return torch.mean(x)
    x = np.array(x)
    return np.mean(x)

function_dict = {
                    0: add, 
                    1: multiply, 
                    2: sin, 
                    3: cos,
                    4: tan,
                    5: sigmoid,
                    6: tanh,
                    7: inverse,
                    8: mean,
                }

d_function_dict = {
                    0: d_add, 
                    1: d_multiply, 
                    2: d_sin, 
                    3: d_cos,
                    4: d_tan,
                    5: d_sigmoid,
                    6: d_tanh,
                    7: d_inverse,
                    8: d_mean,
                }

function_names = {
                    0: 'add', 
                    1: 'multiply', 
                    2: 'sin', 
                    3: 'cos',
                    4: 'tan',
                    5: 'sigmoid',
                    6: 'tanh',
                    7: 'inverse',
                    8: 'mean',
                }

def logger_setup(logger , term_log_level="INFO", file_log_level="DEBUG", log_dir="logs", log_file_name="colonies"):
    logger.remove()
    logger.add(sys.stdout, level=term_log_level)
    #logger.add(f"{log_dir}/{log_file_name}_cants.log", level=file_log_level)
    return logger
