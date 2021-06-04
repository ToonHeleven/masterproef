from tensorflow.keras.layers import GRU, Dropout, Dense, Embedding, Flatten, LSTM, Bidirectional, Input
from tensorflow.keras.models import Sequential
from typing import Union
import numpy as np
from tqdm.keras import TqdmCallback

class BiGRU(Sequential):
    def __init__(self, input_dim, output_dim, seq_length, embedding: Union[None, np.ndarray, str] = None, amntgru: int = 64):
        super().__init__()
        if(type(embedding) == np.ndarray):
            self.add(Embedding(input_dim=input_dim,
                               output_dim=output_dim,
                               weights=[embedding],
                               input_length=seq_length,
                               trainable=False,
                               mask_zero=False,
                               name="embedding"))
        elif(embedding==None):
            self.add(Embedding(input_dim=input_dim,
                               output_dim=output_dim,
                               embeddings_initializer="uniform",
                               input_length=seq_length,
                               trainable=False,
                               mask_zero=False,
                               name="embedding"))
        elif(embedding=="onehot"):
            self.add(Embedding(input_dim=input_dim,
                               output_dim=output_dim,
                               weights=[np.identity(n=input_dim)],
                               input_length=seq_length,
                               trainable=False,
                               mask_zero=False,
                               name="embedding"))

        self.add(Bidirectional(GRU(amntgru, return_sequences=False)))
        self.add(Dropout(0.20))
        self.add(Dense(input_dim, activation="softmax")) #was sigmoid

        print(self.summary())

        self.eventdict = {}
        self.appnamedict = {}
        self.input_dim=input_dim
        self.seq_length = seq_length

    def pretrain(self, trainingset, ttsplit, epochs):
        train_x = np.zeros(shape=(len(trainingset)-10, self.seq_length))
        train_y = np.zeros(shape=(len(trainingset)-10, self.input_dim))
        for i in range(10, len(trainingset)):
            currappname = trainingset.Appname.iloc[i]
            if (self.eventdict.get(currappname) == None):
                if (len(self.eventdict) == 0):
                    for prevappname in trainingset.Appname.iloc[:i]:
                        if (self.eventdict.get(prevappname) == None):
                            newappindex = len(self.eventdict)
                            self.eventdict[prevappname] = newappindex
                            self.appnamedict[newappindex] = prevappname
                currappindex = len(self.eventdict)
                self.eventdict[currappname] = currappindex
                self.appnamedict[currappindex] = currappname

            train_x[i-10,:] = [self.eventdict.get(appname) for appname in trainingset.Appname.iloc[i-10:i]]
            train_y[i-10, self.eventdict.get(trainingset.Appname.iloc[i])] = 1

        trainingsamples = int(ttsplit*len(train_x))
        val_x = train_x[trainingsamples:]
        val_y = train_y[trainingsamples:]
        train_x = train_x[:trainingsamples]
        train_y = train_y[:trainingsamples]
        batchsize = min(len(trainingset), 16)
        self.fit(train_x, train_y, batch_size=batchsize, epochs=epochs, verbose=0,
                 validation_data=(val_x, val_y), callbacks=TqdmCallback())


    def livefit(self, appnames):

        currappname = appnames.iloc[-1]
        if (self.eventdict.get(currappname)==None):
            if (len(self.eventdict) == 0):
                for prevappname in appnames.iloc[:-1]:
                    if (self.eventdict.get(prevappname) == None):
                        newappindex = len(self.eventdict)
                        self.eventdict[prevappname] = newappindex
                        self.appnamedict[newappindex] = prevappname
            currappindex = len(self.eventdict)
            self.eventdict[currappname] = currappindex
            self.appnamedict[currappindex] = currappname

        currapptoken = self.eventdict[currappname]
        prevapptokens = np.array([[self.eventdict.get(prevappname) for prevappname in appnames.iloc[:-1]]])
        currapplabel = np.zeros(shape=(1, self.input_dim))
        currapplabel[0, currapptoken] = 1
        return self.fit(prevapptokens, currapplabel, epochs=4, verbose=0)

    def livefitupdateembedding(self, appnames):

        currappname = appnames.iloc[-1]
        if (self.eventdict.get(currappname) == None):

            if (len(self.eventdict) == 0):
                for prevappname in appnames.iloc[:-1]:
                    if (self.eventdict.get(prevappname) == None):
                        newappindex = len(self.eventdict)
                        self.eventdict[prevappname] = newappindex
                        self.appnamedict[newappindex] = prevappname

            currappindex = len(self.eventdict)
            self.eventdict[currappname] = currappindex
            self.appnamedict[currappindex] = currappname

            prevappindex = self.eventdict.get(appnames.iloc[-2])
            prevembedding = self.get_layer("embedding").get_weights()[0]
            prevembedding[currappindex] = prevembedding[prevappindex]

        currapptoken = self.eventdict[currappname]
        prevapptokens = np.array([[self.eventdict.get(prevappname) for prevappname in appnames.iloc[:-1]]])
        currapplabel = np.zeros(shape=(1, self.input_dim))
        currapplabel[0, currapptoken] = 1
        return self.fit(prevapptokens, currapplabel, epochs=4, verbose=0)

    def livepredict(self, appnames, topk):

        prevapptokens = np.array([[self.eventdict.get(prevappname) for prevappname in appnames.iloc[1:]]])
        prediction = self.predict(prevapptokens, verbose=0)
        return np.array([self.appnamedict.get(appindex) for appindex in (-prediction[0]).argsort()[:topk]])

    def updateembedding(self, newembedding):
        # Divide by zero warning is ok, always returns zero on divide by zero
        newembedding = (np.divide(newembedding.T, newembedding.sum(axis=1), where=newembedding.sum(axis=1)!=0)).T
        self.get_layer("embedding").set_weights([newembedding])
