import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf

#controlling for overscaling
def scaling_dataset(dataframe, fts, oversample = False):
  scaler = StandardScaler()

  features = dataframe[fts].values
  target = dataframe["income"].values

  features = scaler.fit_transform(features)

  #from Kylie Ying @ kylieying.com
  if oversample:
    ros = RandomOverSampler()
    features, target = ros.fit_resample(features, target)

  data = np.hstack((features, np.reshape(target, (-1,1))))

  return data, features, target

def sampling_check(l):
  cases = 0
  controls = 0

  for i in l:
    if i == 1:
      cases += 1
    elif i == 0:
      controls += 1

  return (cases, controls)

#Neural Network
def plot_history(history):
  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,8))
  ax1.plot(history.history['loss'], label = 'loss')
  ax1.plot(history.history['val_loss'],label='val_loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Binary crossentropy')
  ax1.legend()
  ax1.grid(True)

  ax2.plot(history.history['accuracy'], label = 'accuracy')
  ax2.plot(history.history['val_accuracy'],label='val_accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.grid(True)

  plt.show()

def train_model(x_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
  nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(num_nodes,activation='relu',input_shape=(3,)),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(num_nodes,activation='relu'),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(1,activation='sigmoid') #gives an output between 0 and 1, and you can round for classification
  ])

  nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss ='binary_crossentropy', metrics=['accuracy'])

  history = nn_model.fit(
      x_train,y_train,
      epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
  )

  return nn_model, history

ohe = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False).set_output(transform = 'pandas')

#formatting the dataset
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv("adult.data", names = cols)
df.head

#general description of quantitative features
df.describe()

#figuring out missing values
for i in cols:
  unique_values = (i, df[i].unique())
  print(unique_values)
  #? in workclass, ? in occupation, ? in native country

#filtering for missing values in categorical data

df = df.drop(df[df["workclass"] == " ?"].index)
df = df.drop(df[df["occupation"] == " ?"].index)
df = df.drop(df[df["native-country"] == " ?"].index)

#checks
df[df["workclass"] == ' ?'].index
df[df["occupation"] == " ?"].index
df[df["native-country"] == " ?"].index

categorical_names = ['education-num','occupation','workclass', 'native-country', 'education', 'marital-status', 'relationship', 'race']

#changing binary data (income data, sex data) to 1s and 0s for computation
df["income"] = (df["income"] == " >50K").astype(int)
df["sex"] = (df["sex"] == " Female").astype(int)

df["income"]

#counting cases and controls
income_arr = df["income"].values

sampling_check(income_arr)

#there is oversampling of controls

#information for normalization
for label in cols[:-1]:
    plt.hist(df[df["income"]==1][label],label='more than 50k',color='green', alpha=0.6,density=True)
    plt.hist(df[df["income"]==0][label],label='less than 50k',color='red', alpha=0.6,density=True)
    plt.xticks(rotation=90)
    plt.title(label)
    plt.ylabel("probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

#one-hot encoding for categorical data via sci-kit
for i in categorical_names:
  one_hot_encoded = ohe.fit_transform(df[[i]])
  df = pd.concat([df, one_hot_encoded], axis = 1).drop(columns = i)

#splitting into training, validation, and testing datasets
train, valid, test = np.split(df.sample(frac=1),[int(0.6*len(df)), int(0.8*len(df))])

quant_variables = ["age", "fnlwgt", "hours-per-week"]

train, x_train, y_train = scaling_dataset(train, quant_variables, oversample=True)

#mimic real data, so we wouldn't control for oversampling
valid, x_valid, y_valid = scaling_dataset(valid, quant_variables, oversample=False)
test, x_test, y_test = scaling_dataset(test, quant_variables, oversample=False)

sampling_check(y_train)

#training
least_val_loss = float('inf')
least_loss_mdoel = None
epochs = 100

for num_nodes in [16,32,64]:
  for dropout_prob in [0,0.2]:
    for lr in [0.1, 0.005, 0.001]:
      for batch_size in [32, 64, 128]:
        print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
        model,history = train_model(x_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
        plot_history(history)
        val_loss = model.evaluate(x_valid, y_valid)[0]

        if val_loss < least_val_loss:
          least_val_loss = val_loss
          least_loss_model = model