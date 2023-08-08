import pickle
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

data_dict = pickle.load(open('data_v1.pickle', 'rb'))

# print(data_dict.keys())
# print(data_dict)
# df = pd.DataFrame(data_list)

# columns_with_nan = df.isna().any().value_counts()
# print(columns_with_nan)

# Assuming data_dict['data'] is a list of sequences with varying lengths
data_list = data_dict['data']
# Find the maximum length of the sequences in the data
max_length = max(len(sequence) for sequence in data_list)

# Initialize an empty array with the maximum length and a default value (e.g., 0)
data_array = np.zeros((len(data_list), max_length))

# Fill the array with the sequences, padding with zeros for shorter sequences
for i, sequence in enumerate(data_list):
    data_array[i, :len(sequence)] = sequence
# Now 'data_array' will be a 2D NumPy array with consistent shapes for each element

labels = np.asarray(data_dict['labels'])
#print(labels)
x_train, x_test, y_train, y_test = train_test_split(data_array, labels, test_size=0.2, shuffle=True, stratify=labels)
model = SVC()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model_v1.p', 'wb')
pickle.dump({'model_v1': model}, f)
f.close()
