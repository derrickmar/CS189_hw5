import scipy
from scipy import io
import numpy as np
import pdb #  pdb.set_trace()
import matplotlib.pyplot as plt
from DecisionTree import DecisionTree
import csv
import sklearn
import sklearn.utils

# ['Xvalidate', '__globals__', '__header__', 'Ytrain', 'Xtrain', '__version__', 'Yvalidate']]
data = scipy.io.loadmat("../spam-dataset/spam_data.mat")
t_data = sklearn.utils.shuffle(data["training_data"], random_state=0) # (5172, 32)
t_labels = sklearn.utils.shuffle(data["training_labels"].ravel(), random_state=0)  # (1, 5172)
training_data = t_data[0:4137]
training_labels = t_labels[0:4137]
validation_data = t_data[4137:5712]
validation_labels = t_labels[4137:5712]
classifier = DecisionTree()
classifier.train(training_data, training_labels)

error_rate = classifier.test(validation_data, validation_labels)
print error_rate







# TESTING CODE

# predictions = classifier.predict(test_data)
# test_data = data["test_data"] # (5857, 32) last one was 0.46755
# kaggle_results = open("kaggle_spam_or_ham_results.csv", 'wb')
# kaggle_results.truncate()
# wr = csv.writer(kaggle_results)
# wr.writerow(["ID", "Category"])
# for index, prediction in enumerate(predictions):
#   wr.writerow([index+1, prediction])
