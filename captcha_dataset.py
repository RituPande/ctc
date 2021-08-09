# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import PIL
import PIL.Image
import tensorflow as tf
import string
from pathlib import Path
import random

tfds = tf.data.Dataset


class CaptchaData:

    def __init__( self, dir_path, image_height,image_width, batch_size, downsample_factor):
        
        self.downsample_factor = downsample_factor
        self.dir_path = dir_path
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.label_char_to_int = dict()
        self.label_int_to_char  = dict()
        self._create_label_maps()
        converted_labels, file_paths = self._get_label_data(dir_path)
        converted_labels_dataset = tfds.from_tensor_slices(converted_labels)
        image_dataset = self._load_captcha_images(file_paths)
        self.dataset = tfds.zip((image_dataset,converted_labels_dataset)) 
     
       
    def _create_label_maps( self ):
        captacha_nums = '0123456789'
        captcha_chars = string.ascii_lowercase + captacha_nums
        self.label_char_to_int = dict.fromkeys(captcha_chars)
        self.label_int_to_char = dict()
        
        idx = 0
        for  c in self.label_char_to_int:
            self.label_char_to_int[c]  = idx
            self.label_int_to_char[idx] = c
            idx += 1
            
    def _parse_function(self, filename_tensor):
        # create label 
        #label_tensor = tf.strings.split(filename_tensor,sep='/')[-1]
        #label_tensor = tf.strings.split(label_tensor,'.')[0]

        # read image
        img_string_tensor = tf.io.read_file(filename_tensor)
        img_tensor = tf.image.decode_png(img_string_tensor, channels=1)
        #This will convert to float values in [0, 1]
        img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)
        img_tensor = tf.image.resize(img_tensor, [self.image_height, self.image_width])
        #This will transform the image from (img_height, img_width, channel ) to (img_width, img_height, channel)
        img_tensor = tf.transpose(img_tensor, perm=[1, 0, 2])

        return img_tensor

    def _is_test( self, x, y ):
        if x%5 == 0:
            return True
        else:
            return False

    def _is_train( self, x, y ):
        if x%5 != 0:
            return True
        else:
            return False

   
    def _load_captcha_images( self, filepaths ):
        #dataset = tfds.list_files(self.dir_path + "/*.png")
        dataset = tfds.from_tensor_slices(filepaths)
        dataset = dataset.map(self._parse_function, num_parallel_calls=4)
        
        return dataset

    
    def _get_label_data( self, dir_path ):
        
        my_seed = lambda : 0.1
        data_dir = Path(dir_path)
        image_file_names = list(map(str, list(data_dir.glob("*.png"))))
        random.shuffle(image_file_names, my_seed )
        labels = [ file.split('/')[-1].split('.')[0] for file in image_file_names]
        converted_labels = []
       
        for label in labels:
            converted_labels.append(list(map(lambda c:self.label_char_to_int[c] , label )))
                        
        return converted_labels, image_file_names
        
          
    def get_vocab_size( self ):
        return len(self.label_char_to_int)
    
    def split_train_test( self ):
        # keep 20% data in test set, remaining in training set
        recover = lambda x,y: y
        test_dataset  = self.dataset.enumerate().filter( self._is_test ).map(recover)
        train_dataset = self.dataset.enumerate().filter( self._is_train ).map(recover) 
                
        return train_dataset, test_dataset
    
    
    def get_column( self, dataset, column_index ):
        
        column_extractor  = {
            0: lambda x,y: x,
            1: lambda x,y: y
        }
        
        func = column_extractor.get(column_index)
        
        return dataset.map(func)
        
    def make_batches( self, dataset ):
        return dataset.batch(self.batch_size)
    
#IMG_DIR='/kaggle/input/captcha-version-2-images/samples'
#obj = CaptchaData( IMG_DIR, 64,4)
#train, test = obj.split_train_test()
#t = obj.get_column(train,0)
#print(t.as_numpy_iterator().next())










