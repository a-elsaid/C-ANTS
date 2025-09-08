"""
A colony controls:
 - the foraging of ants
 - the creation of RNN from the paths of the ants
 - the training/testing of the RNN
 - updating the pheromone value
 - updating the weights in the search space from the tested RNN
 - evolving the ants charactaristics
 - updating the poplution of the best performing RNN
also use as a PSO particle to evolve coexisting colonies
"""
import sys
from typing import List
from time import time
import warnings
import threading as th
import numpy as np
from sklearn.cluster import DBSCAN
from loguru import logger
from tqdm import tqdm
import torch
from search_space_cants import RNNSearchSpaceCANTS
from search_space_ants import RNNSearchSpaceANTS
from ant_cants import Ant
from rnn import RNN
from timeseries import Timeseries
from helper import Args_Parser
from datetime import datetime
import pickle
from time import time

now = datetime.now()

sys.path.insert(1, "/home/aaevse/loguru")

warnings.filterwarnings("error")
#np.warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)


class Colony:
    """
    Ants Colony
    """

    counter = 1

    def __init__(
        self,
        data: Timeseries,
        logger: logger,
        use_cants: bool,
        use_bp: bool,
        num_epochs: int,
        num_ants: int,
        ants_mortality: int,
        population_size: int,
        max_pheromone: float,
        min_pheromone: float,
        default_pheromone: float,
        space_lags: int,
        dbscan_dist: float,
        evaporation_rate: float,
        log_dir: str,
        out_dir: str,
        num_threads: int,
        search_space=None,
        col_log_level: str = "INFO",
        log_file_name: str = "",
        loss_fun: str = "",
        act_fun: str = "",
    ) -> None:
        self.num_threads = num_threads
        self.logger = logger
        self.out_dir = out_dir
        self.log_file_name = log_file_name
        self.col_log_level = col_log_level
        self.id = self.counter
        self.logger = logger.bind(col_id=self.id)
        self.logger.add(
            f"{log_dir}/{self.log_file_name}_colony_{self.id}.log",
            filter=lambda record: record["extra"].get("col_id") == self.id,
            level=self.col_log_level,
        )
        self.num_epochs = num_epochs
        self.space_lags = space_lags
        self.use_bp = use_bp
        self.log_dir = log_dir
        self.use_cants = use_cants
        if self.use_bp and self.num_epochs == 0:
            self.logger.error("Error: Starting ANTS with 0 Number of Epochs")
            sys.exit()
        self.num_ants = num_ants
        self.best_population_size = population_size
        self.mortality_rate = ants_mortality
        self.evaporation_rate = evaporation_rate
        self.max_pheromone = max_pheromone
        self.min_pheromone = min_pheromone
        self.default_pheromone = default_pheromone
        self.dbscan_dist = dbscan_dist
        self.data = data
        self.loss_fun = loss_fun
        self.act_fun = act_fun
        self.space = search_space
        if not self.space:
            if self.use_cants:
                self.space = RNNSearchSpaceCANTS(
                    inputs_names=data.input_names,
                    outs_names=data.output_names,
                    lags=self.space_lags,
                    logger=self.logger,
                    evaporation_rate=self.evaporation_rate,
                )
            else:
                self.space = RNNSearchSpaceANTS(
                    inputs_names=data.input_names,
                    outs_names=data.output_names,
                    num_hid_nodes=len(data.input_names),
                    num_hid_layers=10,
                    lags=self.space_lags,
                    logger=self.logger,
                    evaporation_rate=self.evaporation_rate,
                )

        self.foragers = [
            Ant(
                ant_id=i + 1,
                logger=self.logger,
                sense_range=np.random.uniform(low=0.1, high=0.9),
                colony_id=self.id,
                log_dir=self.log_dir,
            )
            for i in range(self.num_ants)
        ]
        self.best_rnns = []
        self.pso_position = [self.num_ants, self.mortality_rate, self.evaporation_rate]
        self.pso_velocity = np.random.uniform(
            low=-1, high=1, size=len(self.pso_position)
        )
        self.pso_best_position = self.pso_position
        self.pso_bounds = [[10, 300], [0.01, 0.1], [0.15, 0.95]]
        Colony.counter += 1

    def update_velocity(self, pos_best_g):
        """update new particle velocity"""

        self.logger.info(f"COLONY({self.id}):: Updating Colony PSO velocity")

        w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1  # cognative constant
        c2 = 2  # social constant

        for i, pos in enumerate(self.pso_position):
            r1 = np.random.random()
            r2 = np.random.random()
            vel_cognitive = c1 * r1 * (self.pso_best_position[i] - pos)
            vel_social = c2 * r2 * (pos_best_g[i] - pos)
            self.pso_velocity[i] = w * self.pso_velocity[i] + vel_cognitive + vel_social

    def update_position(self):
        """update the particle position based off new velocity updates"""
        self.logger.info(f"COLONY({self.id}):: Updating Colony PSO position")
        for i, pos in enumerate(self.pso_position):
            self.pso_position[i] = pos + self.pso_velocity[i]

            # adjust position if necessary
            if pos < self.pso_bounds[i][0] or pos > self.pso_bounds[i][1]:
                if i < 1:
                    self.pso_position[i] = np.random.randint(
                        low=self.pso_bounds[i][0], high=self.pso_bounds[i][1]
                    )
                else:
                    self.pso_position[i] = np.random.uniform(
                        low=self.pso_bounds[i][0], high=self.pso_bounds[i][1]
                    )

            self.space.evaporation_rate = self.evaporation_rate

    def forage(
        self,
    ) -> None:
        """
        Letting ants forage to create paths to create RNN
        """

        def ant_thread_work(ant, space):
            ant.reset()
            ant.forage(space)

        # threads = []
        for ant in self.foragers:
            ant.reset()
            ant.forage(self.space)

        """
            threads.append(
                th.Thread(
                    target=ant_thread_work, args=(ant, self.space))
            )
            threads[-1].start()

        for thread in threads:
            thread.join()
        for ant in self.foragers:
            for pnt in ant.new_points:
                self.space.all_points[pnt.id] = pnt
        """

    def calcualte_distance_ceteroid_cluster(
        self,
        point: RNNSearchSpaceCANTS.Point,
        cluster: List[RNNSearchSpaceCANTS.Point],
    ) -> np.ndarray:
        """
        calculates distance between the centeriod of cluster
        and the points in the cluster
        """
        distances = []
        for cluster_point in cluster:
            distance = (
                np.sqrt((point.pos_x - cluster_point.pos_x) ** 2)
                + np.sqrt((point.pos_y - cluster_point.pos_y) ** 2)
                + np.sqrt((point.pos_l - cluster_point.pos_l) ** 2)
                + np.sqrt((point.pos_w - cluster_point.pos_w) ** 2)
            )
            distances.append(
                [
                    distance,
                    cluster_point,
                ]
            )
        if len(distances) < 2:
            return [distances[0][1]]
        try:
            temp = np.array(sorted(distances, key=lambda d: d[0]))[:, 1]
        except np.VisibleDeprecationWarning:
            self.logger.error("Distances list should not be less than 2 elements")
        return temp

    def update_pheromone_const(
        self, rnn: RNN, pheromone_increment: float = 1.0
    ) -> None:
        """
        update the pheromone values using a constant
        """
        for node in rnn.nodes.values():
            point = node.point
            lag_pheromone_boost = (
                1 + float(self.space_lags - point.pos_l) / self.space_lags
            )
            pheromone_increment *= lag_pheromone_boost
            if point.type == 0:
                self.space.inputs_space.increase_pheromone(point, pheromone_increment)
                continue
            if rnn.centeroids_clusters:
                if point.type == 2:
                    self.space.output_space.increase_pheromone(
                        point, pheromone_increment
                    )
                    continue
                point.pheromone = min(
                    point.pheromone + pheromone_increment, self.max_pheromone
                )

                """
                """
                # calculate distance between the centeroid and cluster points
                cluster_distances = self.calcualte_distance_ceteroid_cluster(
                    point, rnn.centeroids_clusters[point.id]
                )
                # distribute pheromone over the cluster points based on distance
                # from ceteroid
                pheromone_step = pheromone_increment / len(cluster_distances)
                for i, pnt in enumerate(rnn.centeroids_clusters[point.id]):
                    pnt.pheromone += pheromone_step * (i + 1) * (1 + pnt.pos_l * 0.1)
                    pnt.pheromone = min(pnt.pheromone, self.max_pheromone)
            else:
                for edge in node.fan_out.values():
                    pnt_link = node.point.fan_out[edge.out_node.point]
                    pnt_link.pheromone = min(
                        pnt_link.pheromone + pheromone_increment, self.max_pheromone
                    )

    def update_pheromone_noreg(
        self,
    ) -> None:
        """
        updating the pheromone values using fitness
        """
        ...

    def update_pheromone_l1reg(
        self,
    ) -> None:
        """
        updating the pheromone values using fitness and L1 regularization
        """
        ...

    def update_pheromone_l2_reg(
        self,
    ) -> None:
        """
        updating the pheromone values using fitness and L2 regularization
        """
        ...

    def update_search_space_weights(self, rnn: RNN) -> None:
        """
        update the weights of the search space from the trained/tested RNN
        """
        with torch.no_grad():
            for node in rnn.nodes.values():
                if len(node.fan_out) == 0:
                    continue
                if rnn.centeroids_clusters:
                    node_edges_avr_wghts = np.average(
                        [e.weight for e in node.fan_out.values()]
                    )
                    for cluster_point in rnn.centeroids_clusters.get(node.point.id, []):
                        cluster_point.pos_w = (
                            cluster_point.pos_w + node_edges_avr_wghts
                        ) / 2
                    node.point.pos_w = (node.point.pos_w + node_edges_avr_wghts) / 2
                else:
                    for edge in node.fan_out.values():
                        point_link = node.point.fan_out[edge.out_node.point]
                        point_link.weight = (point_link.weight + edge.weight) / 2

    def find_centeroid(self, cluster: np.ndarray) -> RNNSearchSpaceCANTS.Point:
        """
        calculate the cenriod of a cluster of points
        used to condensate the points to a centriod
        """
        n = len(cluster)
        sum_x = np.sum(cluster[:, 0])
        sum_y = np.sum(cluster[:, 1])
        sum_l = np.sum(cluster[:, 2])
        sum_w = np.sum(cluster[:, 3])
        new_point = RNNSearchSpaceCANTS.Point(
            sum_x / n, sum_y / n, round(sum_l / n), sum_w / n
        )
        self.space.all_points[new_point.id] = new_point
        return new_point

    def create_nn_ants(
        self,
    ) -> RNN:
        paths = [[] for a in range(self.num_ants)]
        for a in range(self.num_ants):
            cur_pnt = self.space.inputs_space.get_input(ant_exploration_rate=0.5)
            paths[a].append(cur_pnt)
            while cur_pnt.type != 2:
                out_links = list(cur_pnt.fan_out.values())
                pheromones = [link.pheromone for link in out_links]
                norm_pheromones = pheromones / np.sum(pheromones)
                next_link = np.random.choice(out_links, 1, p=norm_pheromones)[0]
                cur_pnt = next_link.out_node
                paths[a].append(cur_pnt)
        return RNN(
            paths=paths,
            centeroids_clusters=None,
            lags=self.space_lags,
            loss_fun=self.loss_fun,
            act_fun=self.act_fun,
        )

    def create_nn_cants(
        self,
    ) -> RNN:
        """
        create RNN from the paths of the ants
        """
        self.forage()
        self.logger.info(f"COLONY({self.id}):: Starting to build RNN from Ants' paths")
        points = []
        for ant in self.foragers:
            for p in ant.path:
                if p.type not in [0, 2]:
                    points.append(p)
        points_vertecies = np.array(
            [[p.pos_x, p.pos_y, p.pos_l, p.pos_w] for p in points]
        )
        db = DBSCAN(
            eps=self.dbscan_dist, min_samples=2, metric="euclidean", n_jobs=-1
        ).fit(points_vertecies)
        labels = db.labels_  # these are the labels for each point in space
        # number of clusters used to iterate over the labels of the clusters
        num_clusters = np.max(db.labels_)
        centeroids = []
        for i in range(num_clusters + 1):
            centeroids.append(self.find_centeroid(points_vertecies[labels == i]))
        # condensed_path: Ant: List[condensed_points]
        condensed_paths = [[] for _ in range(len(self.foragers))]
        counter = 0
        for i, ant in enumerate(self.foragers):
            prev_pnt = ant.path[0]
            for p in ant.path:
                if p.type in [0, 2]:  # checks if the point is for an input or output
                    condensed_paths[i].append(p)
                    continue
                label = labels[counter]
                counter += 1
                if label != -1:  # not single point cluster
                    p = centeroids[label]

                """
                skip point if same-level and before prev_point
                or centroid is same as anther point in path
                """
                if (p.pos_l <= prev_pnt.pos_l and p.pos_y < prev_pnt.pos_y) or (
                    p in condensed_paths[i]
                ):
                    continue

                condensed_paths[i].append(p)
                prev_pnt = condensed_paths[i][-1]
        # clusters will be used to distribute the pheromone
        # on the points in the vicinity of the ceteriods
        clusters = {}
        points = np.array(points)
        labels = np.array(labels)
        for i, point in enumerate(centeroids):
            clusters[point.id] = points[labels == i]
        for i, lag in enumerate(labels):
            if lag == -1:
                clusters[points[i].id] = [points[i]]
        for path in condensed_paths:
            clusters[path[0].id] = [path[0]]
            clusters[path[-1].id] = [path[-1]]

        for i, path in enumerate(condensed_paths):
            self.logger.debug(f"Path {i}")
            for pnt in path:
                self.logger.debug(
                    f"\t Point id: {pnt.id} Point Type: {pnt.type} " +
                    f"Point Name: {pnt.name} x:{pnt.pos_x:.4f} " +
                    f" y:{pnt.pos_y:.4f} l:{pnt.pos_l} w:{pnt.pos_w:.4f}"
                )
        self.logger.info(f"COLONY({self.id}):: Finished building RNN from Ants' paths")
        # self.animate(condensed_paths)
        return RNN(
            paths=condensed_paths,
            centeroids_clusters=clusters,
            lags=self.space_lags,
            loss_fun=self.loss_fun,
        )

    def animate(self, ants_paths) -> None:
        """
        Plot CANTS search space
        """
        points = []
        for level, in_space in enumerate(self.space.inputs_space.inputs_space.values()):
            for pnt in in_space.points:
                points.append([pnt.pos_x, pnt.pos_y, pnt.pos_l, pnt.pheromone])
        for pnt in self.space.output_space.points:
            points.append([pnt.pos_x, pnt.pos_y, pnt.pos_l, pnt.pheromone])
        for pnt in self.space.all_points.values():
            points.append([pnt.pos_x, pnt.pos_y, pnt.pos_l, pnt.pheromone])

        import matplotlib.pyplot as plt

        points = np.array(points)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=points[:, 3] * 10,
            c=points[:, 3],
            cmap="copper",
        )
        for path in ants_paths:
            pnts = []
            for pnt in path[:-1]:
                pnts.append([pnt.pos_x, pnt.pos_y, pnt.pos_l])
            pnts.append([path[-1].pos_x, path[-1].pos_y, self.space.time_lags])
            pnts = np.array(pnts)
            plt.plot(pnts[:, 0], pnts[:, 1], pnts[:, 2])

        plt.show(block=False)
        plt.pause(0.001)
        plt.close()
        plt.cla()
        plt.clf()

    def insert_rnn(self, rnn: RNN) -> None:
        """
        inserts rnn to the population and sorts based on rnns fitnesses
        """
        inserted_rnn = False
        if len(self.best_rnns) < self.best_population_size:
            self.best_rnns.append([rnn.fitness, rnn])
        else:
            if self.best_rnns[-1][0] > rnn.fitness:
                self.best_rnns[-1] = [rnn.fitness, rnn]
                self.update_search_space_weights(rnn)
                self.update_pheromone_const(rnn)
                inserted_rnn = True
                # TODO Put the other pheromone update options

        self.best_rnns = sorted(self.best_rnns, key=lambda r: r[0])
        self.space.evaporate_pheromone()
        self.logger.info(
            f"COLONY({self.id})::\t RNN Fitness: {rnn.fitness:.7f} " +
            f"(Best RNN Fitness: {self.best_rnns[0][0]:.7f})"
        )
        return inserted_rnn

    def evaluate_rnn(self, rnn: RNN) -> None:
        """
        training/testing RNN
        """
        self.logger.info(f"COLONY({self.id}):: Staring RNN Colony Evaluating")

        self.logger.info(f"COLONY({self.id}):: \t starting training")
        if self.use_bp:
            self.logger.info(
                f"\tCOLONY({self.id}):: Using BP, (number of Epochs: {self.num_epochs})"
            )
            # for _ in tqdm(range(self.num_epochs), colour="green"):
            for k in range(self.num_epochs):
                logger.info(f"Evalutating RNN {k}/{self.num_epochs}")
                rnn.do_epoch(
                    self.data.train_input,
                    self.data.train_output,
                    do_feedbck=self.use_bp,
                )
        else:
            rnn.do_epoch(
                self.data.train_input, self.data.train_output, do_feedbck=self.use_bp
            )
        self.logger.info(f"COLONY({self.id}):: \t finished training")

        self.logger.info(f"COLONY({self.id}):: \t starting RNN evaluation")
        rnn.test_rnn(self.data.test_input, self.data.test_output)
        self.logger.info(f"COLONY({self.id}):: \t finished RNN evaluation... Testing Fitness: {rnn.fitness}")

        self.logger.info(f"COLONY({self.id}):: Finished RNN Colony Evaluation")
        return rnn

    def thread_controller(self, total_marchs, num_threads: int):
        from concurrent.futures import ThreadPoolExecutor

        """
        function to control threads for BP CATNS and ANTS
        """
        if self.use_cants:
            logger.info("COLONY({self.id}):: Starting 3D CANTS (with threading)")
        else:
            logger.info("COLONY({self.id}):: Starting ANTS (with threading)")

        def thread_worker(rnn) -> None:
            """
            training/testing RNN
            """
            self.evaluate_rnn(rnn)
            return rnn

        def prepare_rnn():
            if self.use_cants:
                rnn = self.create_nn_cants()
            else:
                rnn = self.create_nn_ants()
            return rnn

        def process_rnn(rnn):
            if rnn.centeroids_clusters:
                for ant in self.foragers:
                    ant.update_best_behaviors(rnn.fitness)
                    ant.evolve_behavior()
            self.insert_rnn(rnn)

        march = 0
        threads = []
        for _ in range(min(total_marchs, num_threads)):
            rnn = prepare_rnn()
            logger.info(f"THREAD {_}")
            executor = ThreadPoolExecutor(max_workers=num_threads)
            feature = executor.submit(thread_worker, rnn)
            threads.append({"thread": executor, "feature": feature})

        turn = True
        while turn:
            if march == total_marchs:
                break
            rnn = prepare_rnn()
            for t in threads:
                if t["feature"].done():
                    process_rnn(t["feature"].result())
                    march += 1
                    logger.info(format(f"March No. {march}", "*^40"))
                    self.logger.info(
                        f"COLONY({self.id}): Interation {march}/{total_marchs}"
                    )
                    if march == total_marchs:
                        turn = False
                        break
                    threads.remove(t)
                    rnn = prepare_rnn()
                    executor = ThreadPoolExecutor(max_workers=1)
                    executor.submit(thread_worker, rnn)
                    threads.append({"thread": executor, "feature": feature})
                    break

    def save_result_to_file(self, stime: float, fitness: float):
        time_stamp = now.strftime('%d_%m_%Y_%H_%M_%S')
        bp = ("bp" if self.use_bp else "wzbp")
        cants = ("CANTS" if self.use_cants else "ANTS")
        with open("/".join([self.out_dir, f"nants{self.num_ants}_{bp}_{cants}.res"]), 'a') as fl:
            fl.write("{}, {}, {}\n".format(time_stamp, stime, fitness))

    def save_rnn(self, rnn: RNN, iteration = ""):
        bp = ("bp" if self.use_bp else "wzbp")
        cants = ("CANTS" if self.use_cants else "ANTS")
        time_stamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        with open("/".join([self.out_dir, f"{iteration}_{time_stamp}_nants{self.num_ants}_{bp}_{cants}.rnn"]), 'wb') as fl:
            pickle.dump(rnn, fl) 

    def life(self, total_marchs) -> None:
        """
        Do colony foraging
        """
        start_time = time()
        if self.num_threads != 0:
            self.thread_controller(total_marchs, self.num_threads)
        else:
            logger.info("COLONY({self.id}):: Starting BP-free 4D CANTS")
            self.num_epochs = 1
            # for march_num in tqdm(range(total_marchs, colour="red")):
            for march_num in range(total_marchs):
                self.logger.info(
                    f"Colony({self.id}): Iteration {march_num}/{total_marchs}"
                )
                rnn = self.create_nn_cants()
                rnn = self.evaluate_rnn(rnn)
                for ant in self.foragers:
                    ant.update_best_behaviors(rnn.fitness)
                    ant.evolve_behavior()
                inserted_rnn = self.insert_rnn(rnn)
                if inserted_rnn:
                    end_time = time() - start_time
                    self.save_rnn(self.best_rnns[0][1], march_num)
            #self.num_epochs = 1

        end_time = time() - start_time
        logger.info(f"Elapsed Time: {end_time}")
        self.save_result_to_file(end_time, self.best_rnns[0][0])



        if self.use_cants:
            colony.use_bp = True
            colony.num_epochs = 1000
            evaluated_rnn = colony.evaluate_rnn(colony.best_rnns[0][1])
            self.save_result_to_file(time()-start_time, evaluated_rnn.fitness)
            self.save_rnn(evaluated_rnn, "-")
            

    def life_mpi(self, total_marchs) -> None:
        """
        Do colony forging with mpi
        """
        from mpi4py import MPI

        mpi_comm = MPI.COMM_WORLD
        mpi_size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        self.logger.info(f"My MPI Rank is : {rank}")

        def worker():
            while True:
                self.logger.debug(f"Worker({rank}) is waiting msg")
                (
                    stop,
                    self.space.all_points,
                    self.space.inputs_space,
                    self.space.output_space,
                ) = mpi_comm.recv(source=0)
                self.logger.debug(f"Worker({rank}) recieved a msg(terminate:{stop})")
                if stop:
                    break

                start_create_nn_time_stamp = time()
                if self.use_cants:
                    rnn = self.create_nn_cants()
                else:
                    rnn = self.create_nn_ants()
                create_nn_time = time() - start_create_nn_time_stamp

                time_log_file = ""
                if self.use_bp:
                    time_log_file = f"cants_wzbp_time_id{rank}.log"
                else:
                    time_log_file = f"cants_wobp_time_id{rank}.log"
                    
                with open(time_log_file, 'a') as f:
                    f.write(f"{create_nn_time}")

                start_eval_nn_time_stamp = time()
                rnn = self.evaluate_rnn(rnn)
                eval_nn_time = time() - start_eval_nn_time_stamp
                with open(time_log_file, 'a') as f:
                    f.write(f",{eval_nn_time}" + "\n")


                for ant in self.foragers:
                    ant.update_best_behaviors(rnn.fitness)
                    ant.evolve_behavior()
                self.logger.debug(f"Worker({rank}) sending a msg")
                mpi_comm.send(
                    [
                        rnn,
                        self.space.all_points,
                        self.space.inputs_space,
                        self.space.output_space,
                    ],
                    dest=0,
                )
                self.logger.debug(f"Worker({rank}) sent a msg")

        def main():
            status = MPI.Status()
            for worker in range(1, mpi_size):
                self.logger.debug(f"Main sending to Worker: {worker}")
                mpi_comm.send(
                    [
                        False,
                        self.space.all_points,
                        self.space.inputs_space,
                        self.space.output_space,
                    ],
                    dest=worker,
                )
                self.logger.debug(f"Main send to Worker:{worker}")
            '''
            for march_num in tqdm(
                range(total_marchs - (mpi_size - 1)),
                colour="red",
                desc="Counting Marchs...",
            ):
            '''
            for march_num in range(total_marchs - (mpi_size - 1)):
                self.logger.info(
                    f"Main Process: Colony({self.id}): Iteration {march_num}/{total_marchs}"
                )
                self.logger.debug("Main waiting for Worker Response")
                (
                    rnn,
                    self.space.all_points,
                    self.space.inputs_space,
                    self.space.output_space,
                ) = mpi_comm.recv(status=status)
                self.logger.debug(f"Main Received from Worker: {status.Get_source()}")
                inserted_rnn = self.insert_rnn(rnn)
                if inserted_rnn:
                    end_time = time() - start_time
                    self.save_rnn(self.best_rnns[0][1], march_num)
                mpi_comm.send(
                    [
                        False,
                        self.space.all_points,
                        self.space.inputs_space,
                        self.space.output_space,
                    ],
                    dest=status.Get_source(),
                )
            '''
            for worker in tqdm(
                range(1, mpi_size),
                colour="red",
                desc=f"Counting Last {mpi_size -1} Marchs...",
            ):
            '''
            for worker in range(1, mpi_size):
                self.logger.info(
                    f"Main Process: Colony({self.id}): Iteration {march_num+worker}/{total_marchs}"
                )
                (
                    rnn,
                    self.space.all_points,
                    self.space.inputs_space,
                    self.space.output_space,
                ) = mpi_comm.recv(status=status)
                inserted_rnn = self.insert_rnn(rnn)
                if inserted_rnn:
                    end_time = time() - start_time
                    self.save_rnn(self.best_rnns[0][1], worker)
                
                mpi_comm.send([True, None, None, None], dest=status.Get_source())

        if rank == 0:
            if colony.use_cants:
                if colony.use_bp:
                    logger.info("Main Process: COLONY({self.id}):: Starting 3D CANTS With-BP")
                else:
                    logger.info("Main Process: COLONY({self.id}):: Starting 4D CANTS BP-Free")
            else:
                logger.info("Main Process: COLONY({self.id}):: Starting ANTS")

                """
                Found that sending the massive structure
                using mpi_comm.send was hitting the max
                allowed number of recurrsive iterations: 1K
                """
                sys.setrecursionlimit(20000)
                logger.info(f"Main Process: Using the total numbe of threads: {sys.getrecursionlimit()}")

            start_time = time()
            main()
            end_time = time() - start_time
            logger.info(f"Elapsed Time: {end_time}")
            self.save_result_to_file(end_time, self.best_rnns[0][0])

            """
            Train only the best rnn with BP
            (ONLY FOR BP-FREE CANTS)
            """
            if colony.use_cants and not colony.use_bp:
                colony.use_bp = True
                colony.num_epochs = 1
                evaluated_rnn = colony.evaluate_rnn(colony.best_rnns[0][1])
                self.save_result_to_file(time()-start_time, evaluated_rnn.fitness)
                self.save_rnn(evaluated_rnn, "-")

        else:
            worker()


if __name__ == "__main__":

    args = Args_Parser(sys.argv)

    logger_format = (
        "\n<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{extra[ip]} {extra[user]} - <level>{message}</level>"
    )
    logger.configure(extra={"ip": "", "user": ""})  # Default values

    logger.remove()
    # logger.add(sys.stdout, level=args.term_log_level)
    logger.add(sys.stdout, format=logger_format, level=args.term_log_level)

    data = Timeseries(
        data_files=args.data_files,
        input_params=args.input_names,
        output_params=args.output_names,
        data_dir=args.data_dir,
        future_time=1,
    )

    colony = Colony(
        data=data,
        num_ants=args.num_ants,
        use_bp=args.use_bp,
        max_pheromone=args.max_pheromone,
        min_pheromone=args.min_pheromone,
        default_pheromone=args.default_pheromone,
        num_epochs=args.bp_epochs,
        population_size=args.colony_population_size,
        space_lags=args.time_lags,
        dbscan_dist=args.dbscan_dist,
        evaporation_rate=args.evaporation_rate,
        log_dir=args.log_dir,
        out_dir=args.out_dir,
        logger=logger,
        col_log_level=args.col_log_level,
        log_file_name=args.log_file_name,
        num_threads=args.num_threads,
        ants_mortality=0.1,
        use_cants=args.use_cants,
        loss_fun=args.loss_fun,
        act_fun=args.act_fun,
    )

    if args.num_threads != 0:
        colony.life(args.living_time)
    else:
        colony.life_mpi(args.living_time)
