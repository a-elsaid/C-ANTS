from ant_cants import Ant
from search_space_cants import SearchSpaceCANTS as Space
import numpy as np
from graph import Graph
from rnn import RNN
from util import function_dict, function_names
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys
from timeseries import Timeseries
import pickle
from typing import List

import multiprocessing as mp
from loguru import logger


from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading

# from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
# import os


def _worker(model, data, cost_type: str, colony_id: int):
        """
        Child process:
        - evaluate the model
        - dump its files (.gv, .eqn, .strct, .png, .model)
        - return (fit, model_filename, pickled_added_pts, pickled_added_in_pts)
        """
        # 1) evaluate
        fit, _ = model.evaluate(data, cost_type=cost_type)

        # 2) dump artifacts (same as before)
        base = f"colony_{colony_id}_model_{model.id}_fit_{fit:.6f}"
        model.visualize_structure(f"{base}.gv")
        model.generate_eqn(f"{base}.eqn")
        model.write_structure(f"{base}.strct")
        model.plot_target_predict(data=data, file_name=f"{base}_target_predict", cost_type=cost_type)
        fname = f"{base}.model"
        with open(fname, "wb") as f:
            pickle.dump(model, f)

        # 3) pickle just the lists of Point objects
        added_pts    = model.get_added_points()     # List[Point]
        added_in_pts = model.get_added_in_points()  # List[Point]
        return float(fit), fname, added_pts, added_in_pts

class Colony():
    count = 0
    def __init__(   self,
                    num_ants: int, 
                    population_size: int, 
                    input_names: List[str], 
                    output_names: List[str],
                    data: Timeseries,
                    num_itrs: int = 10,
                    worker_id: int = None,
                    out_dir: str = "./OUT",
                    use_torch: bool = True,
                    pso_c1:float = 2.05,  # cognative constant
                    pso_c2:float = 2.05,  # social constant
                    cost_type: str = "mse",
                    structure_type: str = "graph",
                    lr=0.001,
    ):

        self.id = Colony.count + 1
        Colony.count += 1
        self.num_itrs = num_itrs
        self.num_ants = num_ants
        self.data = data
        self.best_solutions = []
        self.best_score = None
        self.avg_col_score = None
        self.bst_col_score = None
        self.boost_exploration = True
        self.mortality_rate = np.random.uniform(0.1, 0.5)
        self.evaporation_rate = np.random.uniform(0.1, 0.5)
        self.original_evaporation_rate = self.evaporation_rate  # Store the original evaporation rate
        self.population_size = population_size
        self.life_count = 0
        self.__use_torch = use_torch
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.cost_type = cost_type
        self.structure_type = structure_type
        self.space = Space(
                            input_names=input_names, 
                            output_names=output_names, 
                            evap_rate=self.evaporation_rate
        )
        self.ants = [Ant(self.space) for _ in range(num_ants)]
        self.lr = lr
        
        self.pso_position = [self.num_ants, self.mortality_rate, self.evaporation_rate]
        self.pso_velocity = np.random.uniform(low=-1, high=1, size=len(self.pso_position))
        self.pso_best_position = self.pso_position
        self.pso_bounds = [[5, 20], [0.01, 0.1], [0.15, 0.95]] # Number of ants, mortality rate, evaporation rate
        logger.info(f"Colony({self.id}) (Worker_{worker_id}):: Created with {num_ants} ants and {population_size} population size")

        self._lock = threading.Lock()
        self.out_dir = out_dir

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_lock']  # remove unpicklable lock
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()  # recreate the lock

    def set_evaporation_rate(self, rate):
        self.evaporation_rate = rate
        self.space.evaporation_rate = rate
        
    def check_explored_space(self):
        no_passed = False
        prev_i = 0
        steps = list(np.linspace(0, 1, 5))[1:]
        for i in steps:
            prev_j = 0
            for j in steps:
                prev_k = 0
                for k in steps:
                    prev_l = 0
                    for l in steps:
                        no_passed = True
                        for p in self.space.points:
                            if (
                                (prev_i <= p.get_x() <= i) and 
                                (prev_j <= p.get_y() <= j) and 
                                (prev_k <= p.get_z() <= k) and 
                                (prev_l <= p.get_f() <= l)
                            ):
                                no_passed = False
                                break
                        if no_passed:
                            break
                        prev_l = l
                    if no_passed:
                        break
                    prev_k = k
                if no_passed:
                    break
                prev_j = j
            if no_passed:
                break
            prev_i = i
        return no_passed

    def ants_forage(self, increase_exploration=False):
        with self._lock:    
            paths = []
            for ant in self.ants:
                ant.reset()
                if increase_exploration:
                    ant.explore_rate = 0.999
                ant.march()
                paths.append(ant.path)
                self.space.add_new_points(ant.new_points)
                self.space.add_input_points(ant.new_in_points)
        if self.structure_type == "graph":
            return Graph(
                            ants_paths=paths, 
                            space=self.space, 
                            colony_id=self.id, 
                            use_torch=self.__use_torch,
                            cost_type=self.cost_type,
                        )
        elif self.structure_type == "rnn":
            return RNN(
                            ants_paths=paths, 
                            space=self.space, 
                            colony_id=self.id, 
                            use_torch=self.__use_torch,
                            cost_type=self.cost_type,
                        )
        else:
            logger.error(f"Unknown structure type: {self.structure_type}. Defaulting to Graph.")


    def update_scores(self):
        self.avg_col_score = np.mean([x[0] for x in self.best_solutions])
        self.bst_col_score = self.best_solutions[0][0]



    def insert_to_population(self, score, solution):
        inserted = False
        if len(self.best_solutions) < self.population_size:
            self.best_solutions.append([score, solution])
            inserted = True
        elif score < self.best_solutions[-1][0]:
            self.best_solutions[-1] = [score, solution]
            inserted = True
        self.best_solutions.sort(key=lambda x: x[0])
        return inserted

    def evolve_ants(self, fit):
        for ant in self.ants:
            ant.update_best_behaviors(fit)
            ant.evolve_behavior()

    
    def _dump_model(self, model, fit, cost_type="mse", plot=False):
        
        model.visualize_structure(f"{self.out_dir}/colony_{self.id}_model_{model.id}_fit_{fit}.gv")
        if (False): # fix the condition to work only for Graphs (not RNNS)
            model.generate_eqn(f"{self.out_dir}/colony_{self.id}_model_{model.id}_fit_{fit}.eqn")
        model.save_to_file(f"{self.out_dir}/colony_{self.id}_model_{model.id}_fit_{fit}.model")
        if not plot:
            return
        model.write_structure(f"{self.out_dir}/colony_{self.id}_model_{model.id}_fit_{fit}.strct")
        fig = plt.figure(figsize=(40, 40))
        ax = fig.add_subplot(111, projection='3d')
        # model.plot_path_points(ax=ax, plt=plt)
        model.plot_paths(ax=ax, plt=plt)
        # model.plot_nodes(ax=ax, plt=plt)
        model.plot_pheromones(ax=ax, plt=plt)

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
        plt.savefig(f"{self.out_dir}/colony_{self.id}_model_{model.id}_fit_{fit}.png")
        plt.cla(); plt.clf(); plt.close()
        model.plot_target_predict(data=self.data, file_name=f"{self.out_dir}/colony_{self.id}_model_{model.id}_fit_{fit}_target_predict", cost_type=cost_type)

        fig = plt.figure(figsize=(40, 40))
        ax = fig.add_subplot(111, projection='3d')
        model.plot_nodes(ax=ax, plt=plt)
        plt.savefig(f"{self.out_dir}/colony_{self.id}_nn_{model.id}_fit_{fit}.png")
        plt.cla(); plt.clf(); plt.close()

    def _gen_and_eval(self, increase_exploration: bool, cost_type: str, train_epochs: int = 10):
        """
        Build a model once, evaluate it, return (fit, model).
        """
        thd = threading.current_thread()
        logger.info(f"Colony({self.id:2d}) -- Thread({thd.name}) -- Generating and Evaluating Model")
        model = self.ants_forage(increase_exploration=increase_exploration)


        if not model.single_thrust_test(self.data):
            logger.info(f"Colony({self.id:2d}) -- Thread({thd.name}) -- Single Thrust Test Failed, Regenerating Model") #TODO: Fix nodes not firing issue
            return float('inf'), model
        th_id = threading.current_thread().name
        fit, _ = model.evaluate(self.data, cost_type=cost_type, num_epochs=train_epochs, thread_id=th_id, lr=self.lr)
        return fit, model

    def life_threads(self, num_itrs=None, total_itrs=None, cost_type="mse", train_epochs=10):
        if num_itrs:
            self.num_itrs = num_itrs
        executor = ThreadPoolExecutor(max_workers=num_itrs, thread_name_prefix=f"Colony{self.id:2d}-Thread")
        # each “slot” is one final model you want; we’ll keep trying until it passes
        pending = []           # list of futures
        attempts = dict()      # future -> how many times we’ve tried

        # 1) launch one task per iteration slot
        for slot in range(self.num_itrs):
            fut = executor.submit(self._gen_and_eval, self.boost_exploration, cost_type, train_epochs=train_epochs)
            pending.append(fut)
            attempts[fut] = 1

        done_results = []  # will hold (fit, model) pairs that passed

        # 2) as soon as any future completes, check it:
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                pending.remove(fut)
                count = attempts.pop(fut)
                fit, model = fut.result()

                # if it’s too weak and we haven’t retried 10× yet → re-submit
                if fit > 5 and count < 10:
                    new_fut = executor.submit(self._gen_and_eval, self.boost_exploration, cost_type)
                    pending.append(new_fut)
                    attempts[new_fut] = count + 1
                else:
                    # it’s good (or we’re out of retries)
                    done_results.append((fit, model))

        # 3) now process them in arrival order (or sort by fitness, whatever)
        for fit, model in done_results:
            logger.info(f"Colony({self.id}) - Model({model.id}) fitness={fit:.4e}")

            # *** shared‐state updates under a lock ***
            with self._lock:
                inserted = self.insert_to_population(fit, model)
                self.evolve_ants(fit)
                self.space.evaporate_pheromone(self.evaporation_rate)
                self.space.add_new_points(model.added_points)
                self.space.add_input_points(model.added_in_points)
                if inserted:
                    self.update_scores()
                    self.space.deposited_pheromone(model)
                    self._dump_model(model, fit, cost_type=cost_type)

        # 4) clean up
        for fut in pending:
            # if it’s still running, cancel it
            if not fut.done():
                fut.cancel()
            else:
                # if it’s done, we can still get the result
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"Colony({self.id}): Error in future: {e}")

        executor.shutdown()
    
    def life(self, num_itrs=None, total_itrs=None, cost_type="mse", train_epochs=10):
        if num_itrs:
            self.num_itrs = num_itrs
        patience = 10
        for itr in range(self.num_itrs):
            logger.info(f"Colony({self.id}): Iteration: {self.life_count+1}{f'/{total_itrs}' if total_itrs else ''}")
            self.life_count+=1
            if self.boost_exploration:
                space_is_not_explored = self.check_explored_space()
                if space_is_not_explored:
                    logger.info(f"Colony({self.id}): Space is not fully explored: Boosting Exploration")
                    logger.info(f"Colony({self.id}): Setting Evaporation Rate to 0.999: ON HOLD FOR NOW")
                    # self.evaporation_rate = 0.999
                else:
                    logger.info(f"Colony({self.id}): Space is fully explored: Resetting Exploration")
                    self.boost_exploration = False
                    self.evaporation_rate = self.original_evaporation_rate
            model = self.ants_forage(increase_exploration=self.boost_exploration)

            fit, _ = model.evaluate(self.data, cost_type=cost_type, num_epochs=train_epochs)
            wait = 10
            while fit > 5 and wait > 0:
                wait-=1
                logger.info(f"Colony({self.id}): Fitness is None, ReGenerating & ReEvaluating Model {wait} times left")
                model = self.ants_forage(increase_exploration=self.boost_exploration)
                fit, _ = model.evaluate(self.data)

            logger.info(f"Colony({self.id}) - Model({model.id}): Fitness: {fit}")
            inserted = self.insert_to_population(fit, model)

            self.evolve_ants(fit)
            if inserted:
                self.update_scores()
                self.space.deposited_pheromone(model)
                self._dump_model(model, fit, cost_type=cost_type)

            self.space.add_new_points(model.added_points)
            self.space.add_input_points(model.added_in_points)
            self.space.evaporate_pheromone(self.evaporation_rate)

            '''
            Resting Colony's Evaporation Rate to Max
            if no better models are found
            '''
            if (not self.boost_exploration) and (not inserted):
                logger.info(f"Colony({self.id}): No Improvemnet->Resetting Evaporation Rate")
                patience-=1
            if patience == 0:
                self.boost_exploration = True
                patience = 10
            prev_inserted = inserted


    def get_col_fit(self, rank=None, avg:bool=False) -> float:
        """return the population best fitness"""
        self.update_best_colony_score(rank, avg)
        if avg:
            return self.avg_population_fit, self.pso_best_position
        else:
            return self.bst_col_score, self.pso_best_position



    def update_best_colony_score(self, rank=None, avg:bool=True) -> None:
        best_solutions = np.array(self.best_solutions)
        logger.trace(f"Worker({rank}:: Collecting Fitnees from Colony({self.id})")
        best_scores = best_solutions[:, 0]
        best_score  = np.sort(best_scores)[0]

        '''Get avg of colony fits as measure of overall colony-fit'''
        avg_col_score = sum(best_scores) / len(best_scores) 


        if self.avg_col_score is None or avg_col_score < self.avg_col_score:
            self.avg_col_score = avg_col_score
            if avg:
                self.pso_best_position = self.pso_position

        if self.bst_col_score is None or best_score < self.bst_col_score:
            self.bst_col_score = best_score
            if not avg:
                self.pso_best_position = self.pso_position

    def update_velocity(self, pos_best_g):
        """update new particle velocity"""
        logger.info(f"COLONY({self.id}):: Updating Colony PSO velocity")
        for i, pos in enumerate(self.pso_position):
            r1 = np.random.random()
            r2 = np.random.random()
            vel_cognitive = self.pso_c1 * r1 * (self.pso_best_position[i] - pos)
            vel_social    = self.pso_c2 * r2 * (pos_best_g[i] - pos)
            phi = self.pso_c1 + self.pso_c2
            d = np.abs(2-phi-np.sqrt(phi**2-4*phi))
            if phi<=4:
                logger.error("PSO parameters c1 and c2 do not satisfy the convergence condition (phi > 4).")
                exit(1)
            x = 2 / d  #  Clerc–Kennedy constriction factor
            self.pso_velocity[i] = x*(self.pso_velocity[i] + vel_cognitive + vel_social)

    def update_position(self):
        """update the particle position based off new velocity updates"""
        logger.info(f"COLONY({self.id}):: Updating Colony PSO position")
        
        self.num_ants+= self.pso_velocity[0]
        self.num_ants = int(self.num_ants)
        self.mortality_rate+= self.pso_velocity[1]
        self.evaporation_rate+= self.pso_velocity[2]

        if self.num_ants < self.pso_bounds[0][0] or self.num_ants > self.pso_bounds[0][1]:
            self.num_ants = np.random.randint(
                low=self.pso_bounds[0][0], high=self.pso_bounds[0][1]
            )
            
        if self.mortality_rate < self.pso_bounds[1][0] or self.mortality_rate > self.pso_bounds[1][1]:
            self.mortality_rate = np.random.uniform(
                low=self.pso_bounds[1][0], high=self.pso_bounds[1][1]
            )
            
        if self.evaporation_rate < self.pso_bounds[2][0] or self.evaporation_rate > self.pso_bounds[2][1]:
            self.evaporation_rate = np.random.uniform(
                low=self.pso_bounds[2][0], high=self.pso_bounds[2][1]
            )
            
        self.pso_position[0] = self.num_ants
        self.pso_position[1] = self.mortality_rate
        self.pso_position[2] = self.evaporation_rate

        self.space.evaporation_rate = self.evaporation_rate
        
