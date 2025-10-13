"""
to run the colonies in parallel and evolve them
using PSO
"""
import sys

import pickle
import threading as th
import numpy as np
from loguru import logger
from colony_cants import Colony
from timeseries import Timeseries
from helper import Args_Parser
from search_space_cants import SearchSpaceCANTS as Space
from mpi4py import MPI
from util import logger_setup

comm_mpi = MPI.COMM_WORLD
comm_size = comm_mpi.Get_size()
rank = comm_mpi.Get_rank()

worker_group = np.arange(1,comm_size)
num_colonies = comm_size - 1

class Colonies:
    def __init__(
                self,
                data_files,
                input_params,
                output_params,
                norm_type,
                future_time,
                data_dir,
                living_time,
                out_dir,
                communication_intervals,
                num_epochs=10,
                cost_type="mse",
                pso_c1:float = 2.05,  # cognative constant
                pso_c2:float = 2.05,  # social constant
                structure_type: str = "graph",
                lr=0.001,
    ):
        self.data_files = data_files
        self.input_params = input_params
        self.output_params = output_params
        self.norm_type = norm_type
        self.future_time = future_time
        self.data_dir = data_dir
        self.living_time = living_time
        self.out_dir = out_dir
        self.communication_intervals = communication_intervals
        self.num_epochs = num_epochs
        self.cost_type = cost_type
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.structure_type = structure_type
        self.lr=lr
            
        if rank == 0: # Main Process
            logger.info(f"Main reporting for duty")
        else:   # Worker Process
            logger.info(f"Worker {rank} reporting for duty")
            data = Timeseries(
                        data_files=self.data_files,
                        input_params=self.input_params,
                        output_params=self.output_params,
                        norm_type=self.norm_type,
                        future_time=self.future_time,
                        data_dir=self.data_dir,
                    )

            self.colony = self.create_colony(
                                    data=data, 
                                    living_time=self.living_time, 
                                    out_dir=self.out_dir
                )
            self.colony.id = rank      

            if self.communication_intervals > self.living_time + 1:
                logger.error(f"""
                        Colonies evolution intervals ({self.communication_intervals}) less 
                        than the total number of iterations ({self.living_time+1})""")
                sys.exit()
            

    def create_colony(self, data=None, living_time=None, out_dir="./OUT"):
        num_ants = np.random.randint(low=1, high=20)
        population_size = np.random.randint(low=5, high=25)
        evaporation_rate = np.random.uniform(low=0.7, high=0.9)
        colony = Colony(
                        num_ants=num_ants, 
                        population_size=population_size, 
                        input_names=data.input_names,
                        output_names=data.output_names,
                        data=data,
                        num_itrs=living_time,
                        worker_id=rank,
                        out_dir=out_dir,
                        pso_c1=self.pso_c1,
                        pso_c2=self.pso_c2,
                        cost_type=self.cost_type,
                        structure_type=self.structure_type,
                        lr=self.lr,
                        
                    )
        return colony

    def living_colony(self,):
        """
        used by threads to get the colonies to live in parallel -- WORKERS
        """
        logger.info(f"Starting Colony: Lead Worker({rank}) reporting for duty")

        best_position_global, fitness_global = comm_mpi.recv(source=0)
        logger.info(f"Worker {rank} Received Main's Kickoff Msg")

        for tim in range(self.communication_intervals, self.living_time + 1, self.communication_intervals):
            # colony.life(num_itrs=intervals, total_itrs=living_time, cost_type=cost_type)
            # colony.life_processes(num_itrs=intervals, total_itrs=living_time, cost_type=cost_type)
            self.colony.life_threads(
                                num_itrs=self.communication_intervals, 
                                total_itrs=self.living_time, 
                                cost_type=self.cost_type, 
                                train_epochs=self.num_epochs
            )
            colony_fit, colony_position = self.colony.get_col_fit(rank=rank, avg=False)

            logger.info( 
                            f"Worker({rank}) reporting " +
                            f"its Overall Fitness: {fitness_global:.5f} " +
                            f"for Colony {self.colony.id} " +
                            f"No. Ants ({self.colony.num_ants}) " +
                            f"ER ({self.colony.evaporation_rate:.3f}) " +
                            f"MR ({self.colony.mortality_rate:.3f})  " +
                            f"({tim}/{self.living_time} Living Time)"
            )
            comm_mpi.send((tim, colony_position, colony_fit, rank), dest=0)
            best_position_global, fitness_global = comm_mpi.recv(source=0)
            self.colony.update_velocity(best_position_global)
            self.colony.update_position()
            logger.info(
                            f"\n***>>>===---\n"
                            f"Colony({self.colony.id})::\n" +
                            f"\tBest Global Pos: (Ants:{best_position_global[0]}, MortRate:{best_position_global[1]:.2f}, EvapRate:{best_position_global[2]:.2f})\n" +
                            f"\tBest Col Pos: (Ants:{self.colony.pso_best_position[0]}, MortRate:{self.colony.pso_best_position[1]:.2f}, EvapRate:{self.colony.pso_best_position[2]:.2f})\n" +
                            f"\tNo Ants: {self.colony.num_ants} " +
                            f"\tER: {self.colony.evaporation_rate:.3f}  " +
                            f"\tMR: {self.colony.mortality_rate:.3f}\n" +
                            f"---===<<<***"
            )

        comm_mpi.send(None, dest=0)
        colony_fit, _ = self.colony.get_col_fit(rank=rank, avg=False)
        comm_mpi.send((rank, colony_fit), dest=0)

    def environment(self,):
        '''
        used by main process to manage the colonies -- Maestro (Leader)
        '''
        best_position_global = None
        fitness_global = -1
        BEST_POS_GOL = [0]*num_colonies
        FIT_GOL = np.zeros(num_colonies)
        cols_fits = np.zeros(num_colonies)
        logger.info(f"Main reporting for duty")

        for w in worker_group:
            logger.info(f"Main sending Worker {w} its' kickoff msg") 
            comm_mpi.send((best_position_global, fitness_global), dest=w)
            logger.info(f"Main finished sending Worker {w} its' kickoff msg") 

        done_workers = 0
        while True:
            for c in range(1, num_colonies+1):
                msg = comm_mpi.recv(source=c)
                if msg:
                    tim, best_position, fitness_global, worker_rank = msg
                    BEST_POS_GOL[worker_rank-1] = best_position
                    FIT_GOL[worker_rank-1] = fitness_global
                else:
                    done_workers+=1
                    worker_rank, col_fit = comm_mpi.recv(source=c)
                    cols_fits[worker_rank-1] = col_fit
            if done_workers==num_colonies:
                break
            elif 0<done_workers<num_colonies:
                logger.error("SOMETHING IS WRONG: SOME WORKERS ARE DONE WHILE OTHERS ARE NOT")
                sys.exit()
            fitness_global = np.min(FIT_GOL)
            self.best_position_global = BEST_POS_GOL[np.argmin(FIT_GOL)]
            logger.info(f"*** Finished {tim}/{self.living_time} Living Time ** Best Global Fitness: {fitness_global:.7e} ***")
            for w in worker_group:
                comm_mpi.send((self.best_position_global, fitness_global), dest=w)

        '''
            **** TODO ****
            add code to save the best performing MODEL(STRUCTURE) in each round of intervals    
            this can be done by sending a signal to the lead-worker to save its
            best MODEL(STRUCTURE) if its group did the best job
        '''
        
        best_colony_fit = cols_fits[0]
        for fit in cols_fits[1:]:
            if fit < best_colony_fit:
                best_colony_fit = fit


    def kick_off(self, data = None):
        if rank == 0: # Main Process
            self.environment()
        else:
            if data:
                for k,v in self.colonies.items():
                    v.data = data
            self.living_colony()

def main():    
    """
    main function to run the colonies
    """
    args = Args_Parser(sys.argv)
    intervals = args.communication_intervals
    if intervals > args.living_time + 1:
        logger.error(
            f"Colonies evolution intervals ({intervals}) less" +
            f"than the total number of iterations ({args.living_time+1})"
        )
        sys.exit()
    colonies = Colonies(
                    data_files=args.data_files,
                    input_params=args.input_names,
                    output_params=args.output_names,
                    norm_type=args.normalization,
                    future_time=args.future_time,
                    data_dir=args.data_dir,
                    living_time=args.living_time,
                    out_dir=args.out_dir,
                    communication_intervals=args.communication_intervals,
                    pso_c1=args.pso_c1,
                    pso_c2=args.pso_c2,
                    cost_type=args.loss_fun,
                    structure_type=args.structure_type,
                    num_epochs=args.bp_epochs,
                    lr=args.lr,
    )
    colonies.kick_off()

if __name__ == "__main__":
    main()
