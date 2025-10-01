import numpy as np
from rnn import RNN

def active_inference(self, rnn: RNN, itrs: int = 20) -> (float, float):
    # create a copy of the rnn with random gaussian weights
    rnn.generate_bnn_version()

    print(">>>> Starting Active Inference <<<<")
    self.evaluate_rnn(rnn, active_inference=True)
   

    accumalted_err = []
    for _ in range(itrs):
        accumalted_err.append(rnn.test_rnn(self.data.test_input, self.data.test_output, active_inference=True))
    
    rnn.mean_bnn_fit            = np.mean(accumalted_err, axis=0)
    rnn.uncertianity_prediction = np.std(accumalted_err, axis=0)

    '''Average of RNN fitness, BNN Fitness, and (1- BNN Uncertainty)'''
    rnn.score = np.mean([rnn.fitness, rnn.mean_bnn_fit, 1-rnn.uncertianity_prediction])

    rnn.bnn_nodes.clear()
    del(rnn.bnn_input_nodes)
    del(rnn.bnn_output_nodes)

    print("<<<< Finished Active Inference >>>>")
    
