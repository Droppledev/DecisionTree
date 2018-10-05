from random import seed
from csv import reader
import math
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Load a CSV file
def load_csv(filename):
	file = open(filename, "r")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def get_performance(actual,predicted):
	tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=list(set(actual))).ravel()
	recall = tp/(tp+fn)
	precision = tp/(tp+fp)
	return recall, precision

def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])-1):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	scores = list()
	recall_precision = []
	skf = StratifiedKFold(n_splits=n_folds)
	# Normalize
    # Calculate min and max for each column
	minmax = dataset_minmax(dataset)
	# Normalize columns
	normalize_dataset(dataset, minmax)
	X = []
	classifier = []
	for x in range(len(dataset)):
		X.append(dataset[x][:-1])
		classifier.append(dataset[x][-1])

	for train_index, test_index in skf.split(X,classifier):
		train_set=[]
		test_set=[]
		for i in train_index :
		    train_set.append(dataset[i])
		for i in test_index :
		    test_set.append(dataset[i])
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in test_set]
		accuracy = accuracy_metric(actual, predicted)
		performance = get_performance(actual,predicted)
		recall_precision.append(performance)
		scores.append(accuracy)
	return scores, recall_precision

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Calculate the Entropy for a split dataset
def get_entropy(groups, classes, class_entropy):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gain = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        entropy = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            if p == 0:
                continue
            entropy += (-p * math.log(p,2))
        # weight the group score by its relative size
        gain += entropy * (size / n_instances)
    return class_entropy - gain

# Select the best split point for a dataset
def get_entropy_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    class_entropy = 0
    for class_val in class_values:
        p = [data[-1] for data in dataset].count(class_val) / len(dataset)
        class_entropy += (-p * math.log(p,2))
    b_index, b_value, b_score, b_groups = -1, -1, -1, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gain = get_entropy(groups, class_values, class_entropy)
            if gain > b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gain, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def misclassification_error(groups, classes):
    # count all samples at split point
    n_instances = float(len(classes))
    err = []
    for class_val in classes:
        err.append([row[-1] for row in groups[0]].count(class_val))
    return 1 - ( max(err) / n_instances)

def get_me_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            err = misclassification_error(groups,class_values)
            if err < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], err, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}


# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth, impurity='gini'):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		if impurity == 'gini':
			node['left'] = get_split(left)
		elif impurity == 'entropy':
			node['left'] = get_entropy_split(left)
		elif impurity == 'me':
			node['left'] = get_me_split(left)

		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		if impurity == 'gini':
			node['right'] = get_split(right)
		elif impurity == 'entropy':
			node['right'] = get_entropy_split(right)
		elif impurity == 'me':
			node['right'] = get_me_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size, impurity='gini'):
	root = {}
	if impurity == 'gini':
		root = get_split(train)
	elif impurity == 'entropy':
		root = get_entropy_split(train)
	elif impurity == 'me':
		root = get_me_split(train)

	split(root, max_depth, min_size, 1)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    print_tree(tree)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)

# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))


# load and prepare data
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10

scores, recall_precision = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)

recall = []
precision = []
for i in recall_precision:
	recall.append(i[0])
	precision.append(i[1])

print('Scores: %s' % scores)
print('Recall: %s' % recall)
print('Precision: %s' % precision)
print('Mean Recall: %.3f' % (sum(recall)/float(len(scores))))
print('Mean Precision: %.3f' % (sum(precision)/float(len(scores))))
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))