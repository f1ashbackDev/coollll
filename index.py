import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tensorflow import keras
import math

filename = ('https://github.com/WISEPLAT/SBER-LSTM-Neural-Network-for-Time-Series-Prediction/raw/master/data/SBER_000101_220128.csv')
#filename = ('https://github.com/f1ashbackDev/coollll/SBER_M5.csv')
df = pd.read_csv(filename, sep=",")
print(df.shape, df.columns)
df.rename(columns={"<DATE>":"Date","<TIME>":"Time","<OPEN>":"Open","<HIGH>":"High","<LOW>":"Low","<CLOSE>":"Close","<VOL>":"Volume"}, inplace = True)
print(df.shape, df.columns)
df['Date'] = pd.to_datetime(df['Date'], format = '%Y%m%d')
print(df.shape, df.columns)
df = df.drop('Time', 1)
print(df.shape, df.columns)

today = datetime.today().strftime('%Y-%m-%d')
last_day = df.at[df.shape[0]-1,'Date'].strftime('%Y-%m-%d')
if today == last_day: df = df[:-1]
print(df)

split = 0.85
i_split = int(len(df) * split)
cols = ["Close", "Volume"]
data_train = df.get(cols).values[:i_split]
data_test = df.get(cols).values[i_split:]
len_train = len(data_train)
len_test = len(data_test)
print(len(df), len_train, len_test)

data_train.shape, data_test.shape
((4483,2),(827,2))
sequence_length = 50; input_dim =2; batch_size = 32; epochs = 2

model = tf.keras.Sequential([
	tf.keras.layers.LSTM(100, input_shape=(sequence_length-1, input_dim), return_sequences=True),
	tf.keras.layers.Dropout(.2), 
	tf.keras.layers.LSTM(100, return_sequences=True),
	tf.keras.layers.LSTM(100, return_sequences=False),
	tf.keras.layers.Dropout(.2),
	tf.keras.layers.Dense(1, activation = 'linear')
	])

model = tf.keras.Sequential([
	tf.keras.layers.LSTM(50, input_shape=(sequence_length-1, input_dim), return_sequences=True),
	tf.keras.layers.Dropout(.2), 
	tf.keras.layers.LSTM(50, return_sequences=False),
	tf.keras.layers.Dropout(.2),
	tf.keras.layers.Dense(1, activation = 'linear')
	])

model.summary()
print(model)

model.compile(optimizer='adam',
	loss=tf.keras.losses.MeanSquaredError(),
	metrics=['accuracy'])

model.compile(optimizer='adam',
	loss='mse')

model.compile(optimizer='adam',
	loss='mse',
	metrics=['accuracy'])

def normalise_windows(window_data, single_window=False):
	normalised_data = []
	window_data = [window_data] if single_window else window_data
	for window in window_data:
		normalised_window = []
		for col_i in range(window.shape[1]):
			normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
			normalised_window.append(normalised_col)
		normalised_window = np.array(normalised_window).T
		normalised_data.append(normalised_window)
	return np.array(normalised_data)

def next_window(i, seq_len, normalise):
	window = data_train[i:i+seq_len]
	window = normalise_windows(window, single_window=True)[0] if normalise else window
	x = window[:-1]
	y = window[-1,[0]]
	return x,y

def get_train_data(seq_len, normalise):
	data_x = []
	data_y = []
	for i in range(len_train - seq_len + 1):
		x,y = next_window(i, seq_len, normalise)
		data_x.append(x)
		data_y.append(y)
	return np.array(data_x), np.array(data_y)

x,y = get_train_data(
	seq_len = sequence_length,
	normalise=True
	)

print(x,y,x.shape,y.shape)

def get_train_data2(seq_len, normalise):
	data_window = []
	for i in range(len_train - seq_len + 1):
		data_window.append(data_train[i:i+seq_len])

	data_windows = np.array(data_window).astype(float)
	data_windows = normalise_windows(data_window, single_window=False) if normalise else data_window

	x = data_windows[:, :-1]
	y = data_windows[:, -1, [0]]
	return x,y

x2, y2 = get_train_data2(
	seq_len = sequence_length,
	normalise = True
	)

print("train data shapes: ", x.shape, y.shape)
print("train data shapes: ", x2.shape, y2.shape)

steps_per_epoch = math.ceil((len_train - sequence_length) / batch_size)
print(steps_per_epoch)

batch_size = 32

my_callbacks = [
	tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
	tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)
]

model.fit(x,y, epochs=epochs, batch_size=batch_size, callbacks=my_callbacks)
model.fit(x,y, epochs=epochs)

def get_test_data(seq_len, normalise):
	data_windows = []
	for i in range(len_test - seq_len):
		data_windows.append(data_test[i:i+seq_len])

	data_windows = np.array(data_windows).astype(float)
	data_windows = normalise_windows(data_windows, single_window=False) if normalise else data_windows

	x = data_windows[:, :-1]
	y = data_windows[:, -1, [0]]
	return x,y

x_test, y_test = get_test_data(
		seq_len = sequence_length,
		normalise = True
	)

print("test data shapes: ", x_test.shape, y_test.shape)

model.evaluate(x_test, y_test, verbose = 2)

def get_last_data(seq_len, normalise):
	last_data = data_test[seq_len:]
	data_windows = np.array(last_data).astype(float)
	data_windows = normalise_windows(data_windows, single_window=True) if normalise else data_windows
	return data_windows

last_data_2_predict_prices = get_last_data(-(sequence_length-1), False)
last_data_2_predict_prices_1st_price = last_data_2_predict_prices[0][0]
last_data_2_predict = get_last_data(-(sequence_length-1), True)
print("**** ", -(sequence_length-1), last_data_2_predict.size, "******")

# Урок 6

predictions2 = model.predict(last_data_2_predict)
print(predictions2, predictions2[0][0])

def de_normalise_predicted(price_list, _data):
	return (_data+1) * price_list

def de_normalise_predicted_by_middle(price_max, price_min, _data):
	return (_data*(price_max-price_min) + price_min)

predicted_price = de_normalise_predicted(last_data_2_predict_prices_1st_price, predictions2[0][0])
print(predicted_price)

def predict_point_by_point(data):
	print("[Model] Predicting Point-by-Point....")
	predicted = model.predict(data)
	print("predicted.size =", predicted.size)
	predicted = np.reshape(predicted, (predicted.size,))
	return predicted

predictions2 = predict_point_by_point(last_data_2_predict)
print(predictions2)

def normalise_windows_by_price_by_first_and_volume_by_middle(window_data, single_window = False):
	normalised_data = []
	window_data = [window_data] if single_window else window_data
	for window in window_data:
		normalised_window = []
		for col_i in range(window.shape[1]):
			max_in_column = max(window[:, col_i])
			min_in_column = min(window[:, col_i])
			middle_in_column = (max_in_column + min_in_column) / 2
			if col_i == 0: normalised_col = [((float(p)/ float(window[0, col_i])) -1) for p in window[:, col_i]]
			if col_i == 1: normalised_col = [((float(p)/ float(middle_in_column)) -1) for p in window[:, col_i]]
			normalised_window.append(normalised_col)
		normalised_window = np.array(normalised_window).T
		normalised_data.append(normalised_window)
	return np.array(normalised_data)


df_close = df.loc[:, 'Close']
close_price = df_close.tolist()
plt.plot(close_price)
plt.show

df = df[-365*3:]
print(df)

print(df["Close"].max(), df["Close"].min())
df.shape
df.info()
df.describe()

split = 0.85
i_split = int(len(df) * split)
cols = ["Close", "Volume"]
data_train = df.get(cols).values[:i_split]
data_test = df.get(cols).values[i_split:]
len_train = len(data_train)
len_train = len(data_test)
print(len(df), len_train, len_test)

print(data_train.shape, data_test.shape)

sequence_length = 50; input_dim = 2; batch_size = 32; epochs = 5

def normalise_windows_by_max(window_data, single_window=False):
	normalised_data = []
	window_data = [window_data] if single_window else window_data
	for window in window_data:
		normalised_window = []
		for col_i in range(window.shape[1]):
			max_in_column = max(window[:, col_i])
			normalised_col = [((float(p) / float(max_in_column)) -0) for p in window[:, col_i]]
			normalised_window.append(normalised_col)
		normalised_window = np.array(normalised_window).T
		normalised_data.append(normalised_window)

def normalise_windows_by_middle(window_data, single_window = False):
	normalised_data = []
	window_data = [window_data] if single_window else window_data
	for window in window_data:
		normalised_window = []
		for col_i in range(window.shape[1]):
			max_in_column = max[window[:, col_i]]
			min_in_column = min[window[:, col_i]]
			middle_in_column = (max_in_column - min_in_column) / 2
			sign = -1
			normalised_col = [((float(p) / float(max_in_column)) -0) for p in window[:, col_i]]
			normalised_window.append(normalised_col)
		normalised_window = np.array(normalised_window)
		normalised_data.append(normalised_window)
	return np.array(normalised_data)

def normalise_windows_by_formula(window_data, single_window = False):
	# normalised_data = []
	# window_data = [window_data] if single_window else window_data
	# for window in window_data:
	# 	normalised_window = []
	# 	for col_i in range(window.shape[1]):
	# 		max_in_column = max[window[:, col_i]]
	# 		min_in_column = min[window[:, col_i]]
	# 		normalised_col = [((float(p) / float(max_in_column)) / (float(max_in_column) - float(min_in_column))) for p in window[:, col_i]]
	# 		normalised_window.append(normalised_col)
	# 	normalised_window = np.array(normalised_window)
	# 	normalised_data.append(normalised_window)
	# return np.array(normalised_data)
	normalised_data = []
	window_data = [window_data] if single_window else window_data
	for window in window_data:
		normalised_window = []
		for col_i in range(window.shape[1]):
			max_in_column = max(window[:, col_i])
			min_in_column = min(window[:, col_i])
			middle_in_column = (max_in_column + min_in_column) / 2
			normalised_col = [((float(p) / float(max_in_column)) / (float(max_in_column) - float(min_in_column))) for p in window[:, col_i]]
			normalised_window.append(normalised_col)
		normalised_window = np.array(normalised_window).T
		normalised_data.append(normalised_window)
	return np.array(normalised_data)

def _next_window(i, seq_len, normalise):
	window = data_train[i:i+seq_len]
	window = normalise_windows_by_formula(window, single_window=True)[0] if normalise else window
	x = window[:-1]
	y = window[-1, [0]]
	return x,y

x_train, y_train = get_train_data(
	seq_len=sequence_length,
	normalise=True
)

print(x_train.shape, x_train.shape[0])

for i in range(x_train.shape[0]):
	print(x_train[1])
	break;

prices = x_train[0][:,0]
print(prices)

print(max(prices), min(prices))

volumes = x_train[0][:,1]
print(volumes)
print(min(volumes), max(volumes))
print(x_train, y_train, x_train.shape, y_train.shape)

def get_train_data2(seq_len, normalise):
	data_windows = []
	for i in range(len_train - seq_len + 1):
		data_windows.append(data_train[i:i+seq_len])
	data_windows = np.array(data_windows).astype(float)
	data_windows = normalise_windows_by_formula(data_windows, single_window=True) if normalise else data_windows
	x = data_windows[:, :-1]
	y = data_windows[:, :-1,[0]]
	return x,y

x_train2, y_train2 = get_train_data2(
	seq_len = sequence_length,
	normalise = True
)

def get_train_data(seq_len, normalise):
	data_windows = []
	for i in range(len_train - seq_len + 1):
		data_windows.append(data_train[i:i+seq_len])
	data_windows = np.array(data_windows).astype(float)
	data_windows = normalise_windows_by_formula(data_windows, single_window=True) if normalise else data_windows
	x = data_windows[:, :-1]
	y = data_windows[:, :-1,[0]]
	return x,y

x_train2, y_train2 = get_train_data(
	seq_len = sequence_length,
	normalise = True
)

print("test data shapes: ", x_test.shape, y_test.shape)

steps_per_epoch = math.ceil((len_train - sequence_length)/batch_size)
print(steps_per_epoch)

model = tf.keras.Sequential([
	tf.keras.layers.LSTM(100, input_shape=(sequence_length-1, input_dim), return_sequences=True),
	tf.keras.layers.Dropout(.2), 
	tf.keras.layers.LSTM(100, return_sequences=True),
	tf.keras.layers.LSTM(100, return_sequences=False),
	tf.keras.layers.Dropout(.2),
	tf.keras.layers.Dense(100, activation = 'linear')
])
model.summary()
model.compile(optimizer = "adam", loss="mse", metrics=[tf.keras.metrics.MeanSquaredError()])
callbacks = [
	tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()