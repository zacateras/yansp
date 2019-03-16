import numpy as np
from keras import backend as K

def count_variables(variables):
    return int(np.sum([K.count_params(p) for p in set(variables)]))