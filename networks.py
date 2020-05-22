from tensorflow import keras as kr
from tensorflow.keras import layers
class Mlpnn():
    def __init__(self, arch, obs_shape, act_shape, ):
        #TODO: make custom arch through layers work
        # add batch norm?https://stats.stackexchange.com/questions/304755/pros-and-cons-of-weight-normalization-vs-batch-normalization
        self.model = kr.Sequential()
        self.model.add(layers.Dense(64, activation='relu', input_shape=obs_shape))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(act_shape))



