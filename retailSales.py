import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

feature=pd.read_csv("Features_data_set.csv")
feature['Date']=pd.to_datetime(feature['Date'], format='%d/%m/%Y')

feature['MarkDown1']=feature['MarkDown1'].fillna(0.0)
feature['MarkDown2']=feature['MarkDown2'].fillna(0.0)
feature['MarkDown3']=feature['MarkDown3'].fillna(0.0)
feature['MarkDown4']=feature['MarkDown4'].fillna(0.0)
feature['MarkDown5']=feature['MarkDown5'].fillna(0.0)

feature_cleaned=feature.dropna()

feature_cleaned['year']=feature_cleaned['Date'].dt.year
feature_cleaned['month']=feature_cleaned['Date'].dt.month
feature_cleaned['day']=feature_cleaned['Date'].dt.day
feature_cleaned.drop('Date',axis=1,inplace=True)

feature_copy=feature_cleaned

encoded=LabelEncoder()
feature_cleaned['IsHoliday']=encoded.fit_transform(feature_cleaned['IsHoliday'])

x=feature_cleaned.drop('IsHoliday',axis=1)
y=feature_cleaned['IsHoliday']

x_arr=x.to_numpy()
y_arr=y.to_numpy()

y_new=to_categorical(y_arr)

x_arr_train,x_arr_test,y_new_train,y_new_test=train_test_split(x_arr,y_new,test_size=0.25)

scaler=MinMaxScaler()
scaler.fit(x_arr_train)

x_arr_train_scaler=scaler.transform(x_arr_train)
x_arr_test_scaler=scaler.transform(x_arr_test)

model=Sequential()
model.add(Dense(8,input_dim=13,activation="relu"))
model.add(Dense(8,input_dim=13,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=["accuracy"])

model.fit(x_arr_train_scaler,y_new_train,epochs=150)



