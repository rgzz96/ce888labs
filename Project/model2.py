import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc

train = pd.read_csv('train_audit_encoded.csv')
test = pd.read_csv('test_audit_encoded.csv')

train_y = train['Risk']
train.drop(['Risk'], axis=1, inplace=True)
train_x = train

test_y = test['Risk']
test.drop(['Risk'], axis=1, inplace=True)
test_x = test

train_x =np.array(train_x)
test_x = np.array(test_x)

train_y = np.array(train_y)
test_y = np.array(test_y)

adam = optimizers.adam(lr = 0.005, decay = 0.0000001)

model = Sequential()
model.add(Dense(48, input_dim=train_x.shape[1],
                kernel_initializer='normal',
                activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(24,
                activation="tanh"))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam')

history = model.fit(train_x, train_y, validation_split=0.2, epochs=20, batch_size=64)

predictions_NN_prob = model.predict(test_x)
predictions_NN_prob = predictions_NN_prob[:,0]
predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0)
acc_NN = accuracy_score(test_y, predictions_NN_01)

false_positive_rate, recall, thresholds = roc_curve(test_y, predictions_NN_prob)
roc_auc = auc(false_positive_rate, recall)

print('Overall accuracy of Neural Network model:', acc_NN)
print('AUC Score = '+str(roc_auc))