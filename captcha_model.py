import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, ELU, BatchNormalization, Dense, MaxPooling2D, Reshape
import keras.backend as K
import numpy as np
import sys

        
def loss_func(y_true, y_pred ):
        
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length_batch = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length_batch = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    
    val =  K.ctc_batch_cost(y_true, y_pred, input_length_batch, label_length_batch )
        
    return val
        


class CaptchaModel(Model):
    def __init__( self, num_classes, img_width, img_height ):
        super(CaptchaModel,self ).__init__()
        self.conv1 = Conv2D(16, (3,3), kernel_initializer='he_normal',padding='same',name='Conv1')
        self.bn1 = BatchNormalization()
                
        self.conv2 = Conv2D(32, (3,3), kernel_initializer='he_normal',padding='same',name='Conv2')
        self.bn2 = BatchNormalization()
        self.maxpool2 = MaxPooling2D((2,2), name='pool2')
        
        self.conv3 = Conv2D(64, (3,3), kernel_initializer='he_normal',padding='same',name='Conv3')
        self.bn3 = BatchNormalization()
        self.maxpool3 = MaxPooling2D((2,2), name='pool3')
        
        self.downsample_factor = 4
        new_shape =  ( img_width // self.downsample_factor, (img_height // self.downsample_factor)*64)
        # keras expects input to LTSM in shape=(batchsize, num_timesteps, features)
        # number of timesteps is the width of feature map from the last conv later. 
        # Each timestep is provided all the correspnding columns from all  the  channels concatenated together 
        self.embeddings = Reshape(target_shape=new_shape, name='embeddings')
        self.act = ELU() # this layer is re-used for all convolution activations
              
        self.bltsm1 = Bidirectional(LSTM(128,return_sequences=True, dropout=0.2))
        self.bltsm2 = Bidirectional(LSTM(64,return_sequences=True, dropout=0.2))
        
        self.classifier = Dense(num_classes + 1, activation='softmax', name='output', kernel_initializer='he_normal')
        
    
           
    def call(self, input_tensor ):
       
        x = self.conv1(input_tensor)
        x = self.act(x)
        x = self.bn1(x)
       
        
        x = self.conv2(x)
        x = self.act(x)
        x = self.bn2(x)
        x = self.maxpool2(x) 
                       
        x = self.conv3(x)
        x = self.act(x)
        x = self.bn3(x)
        x = self.maxpool3(x) 
               
        x = self.embeddings(x)
        x = self.bltsm1(x)
        x = self.bltsm2(x)
        
        x = self.classifier(x)
        
        return x
        
class CTCAccuracy(tf.keras.metrics.Metric):
    
    def __init__(self, max_label_len, name ='ctc_accuracy' ):
        super(CTCAccuracy, self).__init__(name)
        self.matches = self.add_weight(name='matches', initializer='zeros')
        self.total   = self.add_weight(name='total', initializer='zeros')
        self.max_label_len = max_label_len
        
        
    def _extract_label( self, x ):
        return tf.slice(x, begin=[0], size=[self.max_label_len] )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        input_len_tensor = input_length * tf.ones(shape=(batch_len,), dtype="int64")
        
        ctc_pred = K.ctc_decode(y_pred, input_length=input_len_tensor, greedy=True)[0][0]
        pred_labels = tf.cast(tf.map_fn(self._extract_label, ctc_pred), y_true.dtype )
  
        values = tf.math.equal(  y_true , pred_labels )
        values = tf.cast(values, tf.float32)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            values = tf.multiply(values, sample_weight)
            
        values = tf.reduce_mean(values, axis = 1)
        result = tf.reduce_sum(values)
        self.matches.assign_add(result)
        self.total.assign_add(tf.cast(batch_len, tf.float32)) 
    
    def reset_state( self ):
        self.matches.assign(0.0)
        self.total(0.0)
        
    def result(self):
        return tf.math.divide_no_nan(self.matches, self.total)
        
        
    
        