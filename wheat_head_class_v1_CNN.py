# -*- coding: utf-8 -*-
"""
Created on Sun May 10 18:09:34 2020

@author: Ratul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from sklearn.model_selection import train_test_split
path='C:/Users/Ratul/Desktop/kaggle/new_folder'

import pandas as pd
import numpy as np
train=pd.read_csv('C:/Users/Ratul/Desktop/kaggle/global-wheat-detection (1)/train.csv')
train.columns
types=train['image_id'].unique()
types_l=types.tolist()

#selecting 10 images for POC
list_types=[]
for i in range(11):
    list_types.append(types_l[i])

list_types_d=pd.DataFrame(list_types,columns=['image_id'])
bbox=pd.merge(list_types_d,train,on='image_id',how='inner')

#new table set for poc ,  based on selected list types
bbox['image_id'].value_counts() 
  
bboximg=[(i+'.jpg') for i in list_types]
for i in bboximg:
    print(i)
bbox.to_csv('C:/Users/Ratul/Desktop/kaggle/global-wheat-detection (1)/train_nw/new_train.csv')


#import images
import os
from os import listdir
import PIL as pl

path='C:/Users/Ratul/Desktop/kaggle/global-wheat-detection (1)/train'
path_nw='C:/Users/Ratul/Desktop/kaggle/global-wheat-detection (1)/train_nw'
for i in listdir(path):
    if i in bboximg:
        x=pl.Image.open(path+'/'+i)
        x.save(path_nw+'/'+i)
        
#sample creation in train_nw folder is complete
    
#testing of image crop

x=pl.Image.open('C:/Users/Ratul/Desktop/kaggle/global-wheat-detection (1)/train_nw/41c0123cc.JPG')
x
area=[47.0, 481.0, 94.0, 86.0]
area_1=(area[0],area[1],area[0]+area[2],area[1]+area[3])
xx=x.crop(area_1)
xx.show()
        



#quick sample crop testing ends

#creating labels
label=[]
for i in listdir(path):
    label.append(i)
 # importing images  
train=[]
for i in listdir(path):
    train.append(plt.imread(path+'/'+i))
    
train[0]

train_arry=np.array(train)
#standardizing training data
train_array_f=train_arry/255

import keras
import tensorflow as tf
from keras.utils import to_categorical

label_unique=set(label)
len(label_unique)
ll=[]
for i in label_unique:
    ll.append(i)
#creating new values for unique levels    
ll_v=[]
for i in range(len(label_unique)):
    ll_v.append(i)
 #   creating the new level dictionary
d=dict(zip(ll,ll_v))
for p in d:
    print(d.get(p))
nw=[]
for k in label:
    if k in d:
        nw.append(d.get(k))

train_label=to_categorical(nw)
train_label=np.array(train_label)
#final test traindataset
x_train,x_test,y_train,y_test=train_test_split(train_array_f,train_label,test_size=0.4)

from keras.models import Sequential
from keras.layers import Flatten,MaxPool2D,Dropout,Conv2D,Dense

model=Sequential()
model.add(Conv2D(16,(3,3),activation='relu',padding='same',input_shape=[1024,1024,3]))
model.add(MaxPool2D((2,2),strides=2,padding='valid'))
model.add(Conv2D(16,(3,3),activation='relu',padding='same'))
model.add(MaxPool2D((2,2),strides=2,padding='valid'))
model.add(Conv2D(16,(3,3),activation='relu',padding='same'))
model.add(MaxPool2D((2,2),strides=2,padding='valid'))
model.add(Flatten())
model.add(Dense(units=112,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=1000)



    
    