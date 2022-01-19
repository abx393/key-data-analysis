import sys
import os
import numpy as np
np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

dir_in = "../native_raw_data_time_series_words"
dir_out = "../features/time_series_words"
keyboard = "HP_Spectre"
model_type = "SVM"

classes = ["the", "why", "ask", "mail", "data", "soon", "wall", "hold", "four", "gold", "thyme", "under",
           "gamma", "pedal", "brake", "climb", "walks", "hello", "world", "smile", "zebra", "space", "rover", "gears", "wheel", "stain",
           "static", "corral", "source", "groggy", "eleven", "quotes", "slight", "reward", "joseph"]

#classes = ["gamma", "pedal", "brake", "climb", "walks", "hello", "world", "smile", "zebra", "space", "rover", "gears", "wheel", "stain"]
features_file = open(os.path.join(dir_out, "time_delays.csv"), 'w')
features_file.write("word,delay1,delay2,delay3,delay4,delay5\n")

x = []
y = []

for f in os.listdir(os.path.join(dir_in, keyboard)):
    (basename, extension) = f.split(".")
    if extension != "csv":
        continue

    with open(os.path.join(dir_in, keyboard, f), 'r') as fr:
        prev_key = ""
        prev_time = 0
        i = 0
        word = ""
        time_delays = []
        for line in fr:
            if i > 0:
                try:
                    [curr_key, curr_time] = line.split(',')
                    word += curr_key
                except:
                    print("i=", i)

                curr_time = float(curr_time)
                time_delay = curr_time - prev_time

                prev_time = curr_time
                prev_key = curr_key

            if i > 1:
                time_delays.append(time_delay)

            i += 1

        # right-pad 'time_delays' with zeroes to length 5
        while len(time_delays) < 5:
            time_delays.append(0)

        if word in classes:
            x.append(time_delays)
            y.append(classes.index(word))
            features_file.write("{},{},{},{},{},{},\n".format(word, time_delays[0], time_delays[1],
                                                          time_delays[2], time_delays[3], time_delays[4]))
features_file.close()

x = np.array(x)

# normalize input by max feature value
x /= np.amax(x)

y = np.array(y)
print("Class distribution:")
print(np.unique(y, return_counts=True))

if model_type != "SVM":
    lb = LabelBinarizer().fit(y)
    y = lb.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

if model_type == "SVM":
    model = SVC()
elif model_type == "NN":
    model = MLPClassifier(max_iter=10000, learning_rate_init=0.01, hidden_layer_sizes=(128,), verbose=True)
elif model_type == "KNN":
    model = KNeighborsClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

if model_type != "SVM":
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", conf_matrix)

for i in range(10):
    print("x_test[i] ", x_test[i])
    print("y_pred[i] ", classes[y_pred[i]])
    print("y_test[i] ", classes[y_test[i]])
    print()

