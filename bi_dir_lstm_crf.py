'''
 Created by Graham Gibson 

 This is a toy example written for proof of concept of architecture design. Currently the architecture is implemented as described  https://arxiv.org/abs/1508.01991

 The basic idea is to connect the representation learned by the Bidir-LSTM to a CRF model, and tweaking the loss function to include both the BD-LSTM outputs and the CRF Transition Matrix as described in the paper. 

 The current example copies binary vectors of length 2, width 2. There is no need to add the CRF since each observation is independent of the others. However, this is an easy example to illustrate the architecture and its ability to learn correctly. 
 
 Both accuracy and example prediction are provided during run time.  


NOTE bleeding edge keras development library required for CRF extension.
TO INSTALL:
  git clone https://github.com/pressrelations/keras
  export PYTHONPATH=$PYTHONPATH:<path-to-cloned-repo>/keras
this will override your default keras install for the current terminal session using PYTHONPATH

'''



import numpy as np
from keras.models import Model
from keras.layers import Input, merge
from keras.layers import Dense, Embedding, ChainCRF, LSTM, Bidirectional, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop



maxlen = 2 
lstm_dim = 10
batch_size = 1
max_features = 100
word_embedding_dim = 10

print('Build model...')

word_input = Input(shape=(maxlen,), name='word_input')
word_emb = Embedding(max_features, word_embedding_dim, input_length=maxlen, dropout=0.2, name='word_emb')(word_input)

bilstm = Bidirectional(LSTM(lstm_dim, dropout_W=0.2, dropout_U=0.2, return_sequences=True))(word_emb)
bilstm_d = Dropout(0.2)(bilstm)
dense = TimeDistributed(Dense(2))(bilstm_d)

crf = ChainCRF()
crf_output = crf(dense)

model = Model(input=[word_input], output=[crf_output])

model.compile(loss=crf.loss,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
y_train = np.zeros((100,2,2),dtype=np.int32)
X_train = []
for sample in range(len(y_train)):
	X_sample = []
	for elm in y_train[sample]:
		flip = np.random.randint(0,2)
		elm[flip] = 1
			
		X_sample.append(flip)
	X_train.append(X_sample)

X_train = np.array(X_train)
print (X_train.shape)
print('Train...')

model.fit(X_train, y_train,
          batch_size=batch_size, nb_epoch=100)

print ("Example")
print (model.predict(X_train)[0])
print (y_train[0])
