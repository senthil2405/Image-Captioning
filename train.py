from dataset import Image_dataset,generate_caption,split_data
from model import master
import config
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import pickle as pkl

gc = generate_caption()
img = Image_dataset('Images/')
data = split_data()




class generate_dataset:
    def __init__(self):
        self.caption_map = {}
        self.features = {}
        self.captions_list = []

    def load_data(self):
        self.features = img.generate_features()

        # Loading Captions
        gc.load_caption('captions.txt')
        self.caption_map = gc.create_map()
        gc.preprocess_caption(self.caption_map)

        captions_list = gc.all_captions()
        self.vocab_size,self.max_len,self.tokenizer = gc.tokenizing()

        # Splitting Data
        self.train_ids,self.test_ids = data.split(self.caption_map)

        return self.features,self.caption_map,self.train_ids,self.test_ids


    def generator(self,ids,caption_map,batch_size):
        
        X1,X2,y = list(),list(),list()
        n = 0 # Number

        while True:
            for id in ids:
                n += 1
                captions = caption_map[id]
                for caption in captions:

                    seq = self.tokenizer.texts_to_sequences([caption])[0]

                    for i in range(len(seq)):
                        input_seq,output_seq = seq[:i],seq[i]

                        input_seq = pad_sequences([input_seq],max_len = self.max_len)
                        output_seq = tf.keras.utils.to_categorical(output_seq,num_classes = self.vocab_size)

                        X1.append(self.features[id])
                        X2.append(input_seq)
                        y.append(output_seq)

                if n == batch_size:
                    X1,X2,y = np.array(X1),np.array(X2),np.array(y)
                    yield [X1,X2],y
                    X1,X2,y = list(),list(),list()
                    n = 0

class train:
    def __init__(self,epochs):
        self.dataset = generate_dataset()
        self.epochs = epochs

    def main(self):
        
        self.features,self.caption_map,self.train_ids,self.test_ids = self.dataset.load_data()

        # Defining optimizer and loss functions
        self.optimizers = tf.keras.optimizers.Adam(lr = 1e-4)
        self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)

        # Define Metrics
        self.train_accuracy = tf.keras.metrics.mean_squared_error(name = 'train_accuracy')
        self.valid_accuracy = tf.keras.metrics.mean_squared_error(name = 'valid_accuracy')

        #Defining model
        self.master_model = master(vocab_size = self.dataset.vocab_size,max_len = dataset.max_len)

        trainloss_per_epoch = []
        validloss_per_epoch = []
        #Calling Training
        for _ in self.epochs:
            print('Training started for epoch {}'.format(_))
            train_loss = self.train_data_for_one_epoch()
            train_acc = self.train_accuracy.result()
            valid_acc = self.valid_accuracy.result()
            valid_loss = self.perform_validation()

            trainloss_per_epoch.append(np.mean(train_loss))
            validloss_per_epoch.append(np.mean(valid_loss))

            print('Epoch no {} : Training Loss : {} and Training Accuracy : {}'.format(_,np.mean(train_loss),train_acc))
            print('Validation Loss : {} , Validation Accuracy : {}'.format(np.mean(valid_loss),valid_acc))

            self.train_accuracy.reset_states()
            self.valid_accuracy.reset_states()






    def apply_gradient(self,optimizer,model,x,y):
        with tf.GradientTape as tape:
            logits = model(x,training = True)
            current_loss = self.loss(y_true = y,y_pred = logits)

        gradients = tape.gradient(current_loss,model.trainable_weights)
        optimizer.apply_gradients(zip(gradients,model.trainable_weights))

        return logits,current_loss

    def train_data_for_one_epoch(self):
        loss = []

        for step,(x_batch_train,y_batch_train) in enumerate(self.dataset.generator(self.train_ids,self.caption_map,batch_size = 10)):
            logits,loss_value = self.apply_gradient(self.optimizers,self.master_model,x_batch_train,y_batch_train)
            loss.append(loss_value)
            self.train_accuracy(logits,y_batch_train)

        return loss

    def perform_validation(self):
        val_losses = []
        for x_val,y_val in self.dataset.generator(self.test_ids,self.caption_map,batch_size = 10):

            val_logits = self.master_model(x_val)
            val_loss = self.loss(y_val,val_logits)
            val_losses.append(val_loss)
            self.valid_accuracy(y_val,val_logits)

        return val_losses

trainer = train(epochs = 1)
trainer.main()





    





