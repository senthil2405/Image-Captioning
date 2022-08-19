import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout,Embedding,LSTM,Add

class vgg_model():
    def __init__(self):

        self.model = VGG16(weights = 'imagenet',include_top = False)

    def create():
        model = Model(inputs = model.inputs,outputs = model.layers[-2].output)
        model.save('vgg16_model.h5')
        return model

class cnn_encoder(Model):
    def __init__(self,units = 256):
        super().__init__()
        self.ce_1 = Dense(units,acitvation = 'relu') 
        self.dropout = Dropout(0.3)



    def call(self,cnn_inputs):
        self.input = cnn_inputs
        x = self.dropout(self.input)
        ce_output = self.ce_1(x)

        return ce_output

class lstm_decoder(Model):
    def __init__(self,vocab_size,embedding_dim,units):
        super().__init__()
        self.embed = Embedding(vocab_size,embedding_dim,mask_zero = True)
        self.lstm = LSTM(units)
        self.dropout = Dropout(0.3)

    def call(self,lstm_inputs):
        self.inputs = lstm_inputs
        x = self.embed(self.inputs)
        embedding = self.embed(x)
        le_out = self.lstm(x)

        return le_out

class master(Model):
    def __init__(self,units,vocab_size,embedding_dim):
        super().__init__()
        self.dense1 = Dense(units,activation = 'relu')
        self.output = Dense(vocab_size,activation = 'softmax')
        self.ce_out = cnn_encoder(units)
        self.le_out = lstm_decoder(vocab_size,embedding_dim,units)

    def call(self,inputs):
        self.cnn_inputs,self.lstm_inputs = inputs
        ce_out = self.ce_out(self.cnn_inputs)
        le_out = self.le_out(self.lstm_inputs)

        x = Add([ce_out,le_out])
        x = self.dense1(x)
        final_output = self.output(x)

        return final_output


    

