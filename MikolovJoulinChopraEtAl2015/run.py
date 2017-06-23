# Structurally Contrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the SCRN model given in Mikolov et al. 2015, arXiv:1412.7753 [cs.NE], 
# https://arxiv.org/abs/1412.7753 using Python and Tensorflow.
#
# This IPython Notebook provides an example of how to call the associated library of Python scripts.  
# Mikolov et al. should be consulted to make sure of the correct hyperparameter values.
#
# Stuart Hagler, 2017

# Imports
import math
import sys
import argparse

# Local Imports
sys.path.insert(0, 'python')
from lstm import lstm_graph
from read_data import read_data
from scrn import scrn_graph
from srn import srn_graph
from tokens import text_elements_to_tokens

# Flags
rnn_flg = 3      # 1 for SRN
                 # 2 for LSTM
                 # 3 for SCRN
usecase_flg = 2  # 1 for predicting letters
                 # 2 for predicting words with cutoff for infrequent words

# Network-specific hyperparameters
if rnn_flg == 1:
    
    # Network hyperparameters
    hidden_size = 110               # Dimension of the hidden vector
    
    # Training hyperparameters
    num_training_unfoldings = 10    # Number of training unfoldings
    
elif rnn_flg == 2:
    
    # Network hyperparameters
    hidden_size = 110               # Dimension of the hidden vector
    
    # Training hyperparameters
    num_training_unfoldings = 10    # Number of training unfoldings
    
elif rnn_flg == 3:
    
    # Network hyperparameters
    alpha = 0.95                    #
    hidden_size = 100               # Dimension of the hidden vector
    state_size = 10                 # Dimension of the state vector

    # Training hyperparameters
    num_training_unfoldings = 50    # Number of training unfoldings
    
# General network hyperparameters
word_frequency_cutoff = 50          # Cutoff for infrequent words for usecase_flg = 2

# General training hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-e', '--num_epochs', type=int,  default=1)
parser.add_argument('-g', '--num_gpus', type=int, default=4)
parser.add_argument('-d', '--data_file', type=str, default="text8.zip")
args = parser.parse_args()

batch_size = args.batch_size        # Batch size for each tower
clip_norm = 1.25                    # Norm for gradient clipping
learning_decay = 1/1.5              # Multiplier to decay the learn rate when required
learning_rate = 0.05                # Initial learning rate
momentum = 0.9                      # Momentum for training
num_epochs = args.num_epochs        # Total number of epochs to run the algorithm
num_gpus = args.num_gpus            # Number of GPUs (towers) available
num_validation_unfoldings = 1000    # Number of validation unfoldings
optimization_frequency = 5          # Number of unfoldings before optimization step
summary_frequency = 500             # Summary information is displayed after training this many batches
filename = args.data_file           # Data file

print("Running with data_file=%s, batch_size=%d, num_epochs=%d, num_gpus=%d" % (filename, batch_size, num_epochs, num_gpus))


# Prepare training, validation, test data sets
raw_data = read_data(usecase_flg, filename)
data, dictionary, reverse_dictionary, vocabulary_size = text_elements_to_tokens(usecase_flg, raw_data, 
                                                                                word_frequency_cutoff)
training_size = int(math.floor((9./11.)*len(raw_data)/num_gpus))
validation_size = int(math.floor((1./11.)*len(raw_data)/num_gpus))
test_size = int(math.floor((1./11.)*len(raw_data)/num_gpus))
training_text = []
validation_text = []
test_text = []
for i in range(num_gpus):
    training_text.append(data[i*training_size:(i+1)*training_size])
    validation_text.append(data[num_gpus*training_size + i*validation_size: \
                                num_gpus*training_size + (i+1)*validation_size])
    test_text.append(data[num_gpus*(training_size + validation_size) + i*test_size: \
                          num_gpus*(training_size + validation_size) + (i+1)*test_size])

print('Vocabulary Size: %d' % vocabulary_size)

# Initiate graph
if rnn_flg == 1:
    # Use SRN
    graph = srn_graph(num_gpus, hidden_size, vocabulary_size, num_training_unfoldings, 
                      optimization_frequency, clip_norm, momentum, batch_size, num_validation_unfoldings)
elif rnn_flg == 2:
    # Use LSTM
    graph = lstm_graph(num_gpus, hidden_size, vocabulary_size, num_training_unfoldings, 
                       optimization_frequency, clip_norm, momentum, batch_size, num_validation_unfoldings)
elif rnn_flg == 3:
    # Use SCRN
    graph = scrn_graph(num_gpus, alpha, hidden_size, state_size, vocabulary_size, num_training_unfoldings, 
                       optimization_frequency, clip_norm, momentum, batch_size, num_validation_unfoldings)
    
# Optimize graph
graph.optimization(learning_rate, learning_decay, num_epochs, summary_frequency, training_text, validation_text)

