import numpy as np 
import pandas as pd
import joblib


# In[12]:


data = pd.read_csv("C:/Users/kisha/OneDrive/Desktop/new ML project/Churn_Modelling.csv")


# In[13]:


data.drop(columns = ['RowNumber','CustomerId','Surname'],inplace=True)


# In[14]:


data = pd.get_dummies(data,columns=['Geography','Gender'],drop_first=True,dtype = int)


# In[15]:


data.head()


# In[17]:


X = data.drop(columns=['Exited'])
y = data['Exited'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[18]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)


# In[20]:


# get_ipython().system('pip install tensorflow')


# In[21]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense


# In[27]:


# model = Sequential()

# model.add(Dense(11,activation='sigmoid',input_dim=11))
# model.add(Dense(11,activation='sigmoid'))
# model.add(Dense(1,activation='sigmoid'))
# model = Sequential()
# model.add(Dense(11, activation='relu', input_dim=11))   # Hidden Layer 1
# model.add(Dense(11, activation='relu'))                 # Hidden Layer 2
# model.add(Dense(1, activation='sigmoid'))               # Output Layer
# model = Sequential()

# model.add(Dense(11,activation='sigmoid',input_dim=11))
# model.add(Dense(11,activation='sigmoid'))
# model.add(Dense(1,activation='sigmoid'))
model = Sequential()
model.add(Dense(11, activation='relu', input_dim=11))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# In[28]:


model.summary()


# In[29]:


model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[30]:

from sklearn.utils import class_weight
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
model.fit(X_train, y_train, batch_size=50, epochs=100, validation_split=0.2, class_weight={0: weights[0], 1: weights[1]})
# history = model.fit(X_train,y_train,batch_size=50,epochs=100,verbose=1,validation_split=0.2)

# model.save("churn_model.h5")  # ✅ Save model
# joblib.dump(scaler, "scaler.save")  # ✅ Save scaler
# model.save("churn_model.h5")
# joblib.dump(scaler, "scaler.save")



# In[31]:


# y_pred = model.predict(X_test)
# In[32]
# y_pred = model.predict(X_test_trf)
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=-1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)