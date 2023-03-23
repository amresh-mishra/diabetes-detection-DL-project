from numpy import loadtxt 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping


# loading the dataset
dataset = loadtxt ('pima-indians-diabetes.csv', delimiter=',')
x=dataset[:,0:8]
y=dataset[:,8]




#Scaling the dataset

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X=sc.fit_transform(x)

#spliting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=20,random_state=1)

# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss',min_delta=0.001,restore_best_weights=True, patience=10, verbose=1, mode='auto')

#model builiing 
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # 12 neuron on first and 8 on second layer
model.add(Dense(8, activation='relu')) 
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))  # last layer showing person is diabitic or not 


#compliation 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
'''from keras.optimizers import SGD
model.compile(SGD(lr=0.003), loss='binary_crossentropy',   metrics=['accuracy'])'''


history=model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=1000,   batch_size= 10, callbacks=[early_stop])  # used early EarlyStopping

#accuracy check
_, accuracy = model .evaluate(X_test, y_test) 
print ("Accuracy : % .2f" % (accuracy * 100))

predictions = (model.predict(X_test) > 0.5).astype("int32")

for i in range(5,10):
	print('Predicted Class: %d (Original Class: %d)' % (predictions[i], y[i]))

import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()





model_json = model.to_json()
with open ("model.json", 'w') as json_file :
    json_file.write(model_json)
model.save_weights ('model.h5')
print("saved model to disk")
