import os
from torch import cuda

MODEL_NAME = 'google/bigbird-roberta-base'
config = {'model_name': MODEL_NAME,
         'max_length': 1024,
         'train_batch_size':4,
         'valid_batch_size':4,
         'epochs':5,
         'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
         'max_grad_norm':10,
         'device': 'cuda' if cuda.is_available() else 'cpu'}

# THIS WILL COMPUTE VAL SCORE DURING COMMIT BUT NOT DURING SUBMIT
COMPUTE_VAL_SCORE = True
if len( os.listdir('./input/feedback-prize-2021/test') )>5:
      COMPUTE_VAL_SCORE = False