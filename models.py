import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ReLU, Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization, LSTM, Dropout, Reshape, TimeDistributed, add, Bidirectional
from tensorflow.keras import Model, Sequential
from keras.regularizers import l2

class melody_extraction(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(filters=64,name='conv1',kernel_size=(5,5),input_shape=(500,513),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        self.bn1 = BatchNormalization(name='bn1')
        self.conv2 = Conv2D(filters=128,name='conv2',kernel_size=(5,5),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        self.bn2 = BatchNormalization(name='bn2')
        self.conv3 = Conv2D(filters=192,name='conv3',kernel_size=(5,5),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        self.bn3 = BatchNormalization(name='bn3')
        self.conv4 = Conv2D(filters=256,name='conv4',kernel_size=(5,5),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        self.bn4 = BatchNormalization(name='bn4')
        self.linear1 = Dense(512,name='dense1',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        self.final = TimeDistributed(Dense(506,name='dense2',activation='softmax',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5)))
        

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.keras.activations.relu(x)
        x = MaxPooling2D((1,4))(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.keras.activations.relu(x)
        x = MaxPooling2D((1,4))(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.keras.activations.relu(x)
        x = MaxPooling2D((1,4))(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.keras.activations.relu(x)
        x = MaxPooling2D((1,4))(x)

        x = Reshape((500,x.shape[2]*x.shape[3]))(x)     
        int_output = self.linear1(x)
        x = self.final(int_output)
        return x,int_output

    def build_graph(self,raw_shape):
        x = Input(shape=raw_shape)
        return Model(inputs=[x], outputs = self.call(x)) 


class ConfidenceModel(Model):
    def __init__(self,fe_model):
        super().__init__()
        self.pretrain = fe_model     
        self.dense2 = Dense(256,activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        self.final = Dense(1,activation='sigmoid',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        
    def call(self,x):
        _,x = self.pretrain(x)     
        
        x = self.dense2(x)
        x = Dropout(0.2)(x)
        x = self.final(x)
        return x
    
    def build_graph(self,raw_shape):
        x = Input(shape=raw_shape)
        return Model(inputs=[x],outputs = self.call(x))