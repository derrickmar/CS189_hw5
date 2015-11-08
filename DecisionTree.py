import scipy
from scipy import io
import numpy as np
import pdb #  pdb.set_trace()
from collections import Counter
from Node import Node
from math import log
import inspect


# # 1 means spam, 0 means ham
class DecisionTree:
  def __init__(self):
    self.root_node = None
    self.max_depth = 20
    self.predictions = []

  # { 0: 100, 1: 200}
  def impurity(self, left_label_hist, right_label_hist):
    left_total = sum(left_label_hist.values())
    right_total = sum(right_label_hist.values())
    total = left_total + right_total

    probability_left = left_total / float(total)
    entropy_left = self.entropy(left_label_hist)
    probability_right = right_total / float(total)
    entropy_right = self.entropy(right_label_hist)

    parent_entropy = self.entropy(Counter(left_label_hist) + Counter(right_label_hist))
    return parent_entropy - (entropy_left*probability_left + entropy_right*probability_right)

  def entropy(self, histogram):
    total = sum(histogram.values())
    if total == 0:
      return 1

    result = 0
    for value in histogram.values():
      proportion = value / float(total)
      if proportion == 0:
        continue
      result = result - (proportion*log(proportion, 2))
    return result

  def segmentor(self, train_data, train_labels):
    split_rule = None
    max_information_gain = float("-inf")
    all_feature_arrays = train_data.T

    for feature_array in all_feature_arrays:
      # TODO: Sort each feature array
      # TODO: Might want to create more thresholds
      thresholds = self.calculateThresholds(feature_array)
      left_label_histogram = { 0: 0, 1: 0 }
      right_label_histogram = { 0: 0, 1: 0 }
      for threshold in thresholds:
        curr_feature = 0
        for index, feature_value in enumerate(feature_array):
          label_for_feature_value = train_labels[index]
          if feature_value < threshold:
            left_label_histogram[label_for_feature_value] = left_label_histogram[label_for_feature_value] + 1
          else:
            right_label_histogram[label_for_feature_value] = right_label_histogram[label_for_feature_value] + 1

        information_gain_split = self.impurity(left_label_histogram, right_label_histogram)
        if information_gain_split > max_information_gain:
          max_information_gain = information_gain_split
          split_rule = (curr_feature, threshold)

        curr_feature += 1

    print max_information_gain
    if max_information_gain <= 0:
      return None

    # Generate the train_data and train_labels so we can pass it to next node
    left_split_data = []
    left_split_labels = []
    right_split_data = []
    right_split_labels = []
    # Now we have the optimal split rule based on lowest impurity
    optimal_split_feature = split_rule[0]
    optimal_threshold = split_rule[1]
    for index, feature_value in enumerate(all_feature_arrays[optimal_split_feature]):
      label = train_labels[index]
      sample = train_data[index]
      if feature_value < optimal_threshold:
        left_split_data.append(sample)
        left_split_labels.append(label)
      else:
        right_split_data.append(sample)
        right_split_labels.append(label)

    return {
      "split_rule": split_rule,
      "left_split": {
        "train_data": np.array(left_split_data),
        "train_labels": np.array(left_split_labels)
      },
      "right_split": {
        "train_data": np.array(right_split_data),
        "train_labels": np.array(right_split_labels)
      }
    }

  ## Calculate majority value for labels
  def majorityValue(self, train_labels):
    majority_value = 0
    num_non_zeros = np.count_nonzero(train_labels)
    if num_non_zeros > (1 - num_non_zeros):
      majority_value = 1
    return majority_value

  def growTree(self, train_data, train_labels, curr_depth):
    print "growTree"
    if curr_depth > self.max_depth:
      # Calculate majority labels and this is your
      majority_value = 0
      num_non_zeros = np.count_nonzero(train_labels)
      if num_non_zeros > (1 - num_non_zeros):
        majority_value = 1
      return Node(None, None, None, majority_value) # leaf node

    if np.all(train_labels == 0): # all ham
      return Node(None, None, None, 0)

    if np.all(train_labels == 1): # all spam
      return Node(None, None, None, 1)

    segmentation = self.segmentor(train_data, train_labels)
    # If the information gain is not any better end with leaf node
    if segmentation is None:
      majority_value = self.majorityValue(train_labels)
      return Node(None, None, None, majority_value)


    # pdb.set_trace()
    left_split = segmentation["left_split"]
    right_split = segmentation["right_split"]
    return Node(
      segmentation["split_rule"],
      self.growTree(left_split["train_data"], left_split["train_labels"], curr_depth + 1),
      self.growTree(right_split["train_data"], right_split["train_labels"], curr_depth + 1)
    )

  def train(self, train_data, train_labels, curr_depth = 0):
    self.root_node = self.growTree(train_data, train_labels, curr_depth)

  def predict(self, test_data):
    predictions = []
    for sample in test_data:
      predictions.append(self.predict_sample(sample))
    return predictions

  def test(self, validation_data, validation_labels):
    predictions = np.array(self.predict(validation_data))
    difference = predictions - validation_labels
    error_rate = np.count_nonzero(difference) / float(len(validation_labels))
    return error_rate

  def predict_sample(self, sample):
    curr_node = self.root_node
    while isinstance(curr_node, Node):
      curr_node = curr_node.next_node(sample)

    return curr_node

  def calculateThresholds(self, feature_values):
    return np.unique(feature_values)
    # threshold = np.sum(feature_values) / float(len(feature_values))
    # print threshold
    # return [threshold]

