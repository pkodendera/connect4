
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
# import tensorflow as tf
# from tensorflow import keras
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import  losses, metrics
from sklearn.model_selection import train_test_split


# In[5]:


df = pd.read_table("connect-4.data", sep=',', header=None)
cols = list(range(42))
cols.append("label")
df.columns = cols

# b represents a blank spot. will be 0
# x represents a play by x. will be 1
# o represents a plya by o. will be 2
# win/loss in persepective of x

replace_dict = {"b": "0", "x": "1", "o": "2", "loss":"0", "draw":"1", "win":"2"}
df.replace(to_replace=replace_dict, inplace=True)
df.tail()

#split data into training and test sets
# split_index = (int)(len(df) * .8)
# train_df = df[0:split_index]
# test_df = df[split_index:]


# In[7]:


# tf_dataset = (
#         tf.data.Dataset.from_tensor_slices(
#             (
#                 tf.cast(df[cols[:-1]].values, tf.int32),
#                 tf.cast(df['label'].values, tf.string)
#             )
#         )
# )
# print (tf_dataset)
# iter = tf_dataset.make_one_shot_iterator()
# el = iter.get_next()
# with tf.Session() as sess:
#     print(sess.run(el))


# In[8]:


np_dataset = df.values
np_dataset.shape


# In[9]:


seed = 10
np.random.seed(seed)
X = np_dataset[:,:-1]
Y = np_dataset[:,-1]
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.3, random_state=seed)


# In[10]:


model = Sequential()
model.add(Dense(42,input_dim=42, kernel_initializer='uniform', activation='relu'))
model.add(Dense(29, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='selu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


# In[11]:


model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, batch_size=200, verbose=0)
scores = model.evaluate(X_test, Y_test)
print("Accuracy:", (scores[1]*100))


# In[ ]:


df_loss = df[df["label"]=="0"]
print (len(df_loss))
df_draw = df[df["label"]=="1"]
print (len(df_draw))
df_win = df[df["label"]=="2"]
print (len(df_win))


# In[ ]:


get_ipython().system('jupyter nbconvert --to script co.ipynb')

