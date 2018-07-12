import pandas as pd
filename='G:\\python_work\\sales_data.xls'
data=pd.read_excel(filename,index_col=u'序号')
data[data==u'好']=1
data[data==u'高']=1
data[data==u'是']=1
data[data!=1]=0
data=data.as_matrix()
data_x=data[:,:3]
data_y=data[:,3]
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from sklearn.metrics import confusion_matrix
model=Sequential()
model.add(Dense(input_dim=3,output_dim = 10))
model.add(Activation('relu'))
model.add(Dense(input_dim=10,output_dim = 1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(data_x,data_y,nb_epoch=1000,batch_size=1)
yp=model.predict_classes(data_x).reshape(len(data_y))
cm=confusion_matrix(list(data_y),list(yp))
import matplotlib.pyplot as plt
plt.matshow(cm,cmap=plt.cm.Greens)
plt.colorbar()
for x in range(len(cm)):
    for y in range(len(cm)):
        plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()














