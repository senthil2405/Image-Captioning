import os
import zipfile
import tensorflow as tf
import keras.backend as K
import numpy as np
import pickle as pkl

from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from model import vgg_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer



class Image_dataset():
    def __init__(self,image_path):
        self.file_path = image_path
        self.features = {}
        
        
    
    def generate_features(self):
        list_ds = tf.data.Dataset.list_files(self.file_path+ '*.jpg',shuffle = False)

        model = vgg_model()


        for img_path in list_ds:

            img = tf.io.read_file(np.array(img_path))
            img = tf.image.decode_jpeg(img,channels = 3)

            img = tf.image.resize(img,(160,160),preserve_aspect_ratio=True)
            #img = tf.image.reshape(img,[1,img.shape[0],img.shape[1],img.shape[2]])

            img = np.array(img).reshape([1,img.shape[0],img.shape[1],img.shape[2]])
            feature = np.array(preprocess_input(img))

            image_id = tf.strings.split(img_path,os.path.sep)[-1]
            self.features[str(image_id.numpy())] = feature


        pkl.dump(self.features,open('features.pkl','wb'))

        return self.features


class generate_caption():
    def __init__(self):
        self.caption_map = {}
        self.captions = []
        self.captions_list = [] #Preprocessed Tagged Captions List
        self.tokenizer = Tokenizer()

    def load_caption(self,path):
        with open(path,'rb') as f:
            self.captions = f.read()

    def create_map(self):

        for caption in self.captions.split('\n'):
            caption_id,line = caption.split(',')[0],caption.split(',')[1:]

            if len(line) >2:
                continue
            
            # Removing .jpg from captions
            caption_id = caption_id.split('.')[0]

            # Converting caption id to list
            line = ''.join(line)
            if caption_id not in self.caption_map:
                self.caption_map[caption_id] = []

            self.caption_map[caption_id].append(line)

            print(len(self.caption_map))

        return self.caption_map



    def print_caption(self):
        for id,caption in self.caption_map.items():
            print(id)
            print('\n',caption)
            break

    def preprocess_caption(caption_map):
        ## Function to clean the captions and add start and endtags
        ## When we access caption_map.items() we get a referrence to the actual dict
        ## therfore if we write captions[i] = caption it gets affected at captionmap

        for id,captions in caption_map.items():
            for i in range(len(captions)):
                caption = captions[i]

                caption = caption[i]
                caption = caption.lower()
                caption = caption.replace('[A-Za-z]','')

                caption = caption.replace('/s+',' ')

                # Adding start and end tags
                # Here we use join to convert caption into string for concatenating

                caption = 'starttag ' + ''.join([word for word in caption.split()]) + ' endtag'
                captions[i] = caption

    def all_captions(self):
        for key in self.caption_map.keys():
            for caption in self.caption_map[key]:
                self.captions_list.append(caption)
        print(len(self.captions_list))
        return self.captions_list

    def tokenizing(self):
        self.tokenizer.fit_on_texts(self.captions_list)
        vocab_size = len(self.tokenizer.word_index)+1

        max_len = max(len(caption.split()) for caption in self.captions_list)


        pkl.dump(self.tokenizer,open('tokenizer.pkl','wb'))

        return vocab_size,max_len,self.tokenizer

class split_data:
    def __init__(self):
        self.train = []
        self.test = []

    def split(self,caption_map):
        caption_id = list(caption_map.keys())

        split = int(len(caption_id)*0.96)

        self.train = caption_id[:split]
        self.test = caption_id[split:]

        return self.train,self.test


    



    





    





        

            

        
