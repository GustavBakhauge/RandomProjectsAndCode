import numpy as np

#Helper class for constructing tree
class Node:
    def __init__(self, gini, samples, samples_pr_class, predict_class):
        self.gini = gini
        self.samples = samples
        self.samples_pr_class = samples_pr_class
        self.predict_class = predict_class
        self.feature_i = 0
        self.split_value = 0
        self.left = None
        self.right = None


#Decision Tree Class
class DescTree: 
    def __init__(self, max_depth = 10):
        """Takes a max_depth hyperparameter to ensure no overfitting happens"""
        self.max_depth = max_depth #max depth of tree
        self.target_val = 0 #target value that bool_check compares to
        self.best_q = None #best value to split the dataset on
        self.best_feature = None #best feature to split the dataset in

    def bool_check(self, test_val):
        """Checks if a value is greater or equal to a target value, returns True/False"""
        return test_val >= self.target_val

    def split(self, X):
        """Splits data into left and right arrays (True/False) on a condition checked in our bool_check function """
        left_data, right_data = [], []
        for row in X:
            if self.bool_check(row) == True:
                left_data.append(row)
            else: 
                right_data.append(row)
        return left_data, right_data

    def calculate_gini(self,y):
        """ Calculates a gini_score """
        classes, count = np.unique(y, return_counts=True)
        gini = 1 
        for p in range(len(classes)):
            pri = count[p]/np.sum(count)
            gini -= pri**2
        return gini
        
    def info_gain(self, left_data, right_data, gini):
        """Calculates an info_gain_score by calling the calculate_gini function"""
        prior = float(len(left_data)) / (len(left_data) + len(right_data))
        info = gini - (prior * self.calculate_gini(left_data)) - ((1-prior)* self.calculate_gini(right_data))
        return info

    def best_split(self, X,y):
        """Finds the best value to split on by itterating through each row in each feature, 
        saves the best info gain, best value and best feature while itterating and 
        returns the best feature + value that gives the highest info gain.  """
        best_info_gain = -10 
        best_bool_check = None
        best_feature = None
        rows, features = X.shape
        for feature in range(features):
            unique = np.unique(X[:,feature])
            for u in unique:
                impurity = self.calculate_gini(y)
                self.target_val = u
                LD, RD = self.split(X[:,feature])
                if len(LD) == 0 or len(RD) == 0:
                    continue
                info = self.info_gain(LD, RD, impurity)
                if info >= best_info_gain:  
                    best_info_gain, best_bool_check, best_feature = info, self.target_val, feature

        self.best_feature = best_feature
        self.best_q = best_bool_check
        return self.best_q, self.best_feature, best_info_gain

    def fit(self, X, y):
        """Fits the data, calls all our functions inside this class"""
        self.n_features = X.shape[1] 
        self.n_classes = len(set(y)) 
        self.tree = self.create_tree(X,y)

    def create_tree(self, X, y, depth = 0):
        """ A recursive function that creates a decision tree"""
        cl , samples_pr_class = np.unique(y, return_counts=True)
        yL = list(y)
        most_class = max(set(yL), key=yL.count)
        node = Node(gini = self.calculate_gini(y), samples = y.size, samples_pr_class = samples_pr_class, predict_class = most_class)
        #Making use of depth parameter
        if depth < self.max_depth:
            value, feat, info_gain = self.best_split(X,y)
            if feat != None:
                index = X[:,feat] < value
                X_left, y_left = X[index], y[index]
                X_right, y_right = X[~index], y[~index]
                node.feature_i = feat
                node.split_value = value 
                node.left = self.create_tree(X_left, y_left, depth + 1)
                node.right = self.create_tree(X_right, y_right, depth +1)
        return node

    def predict(self, X):
        """Calls another function that predicts the class for each row in a test dataset """
        predictions = []
        for values in X:
            node = self.tree
            while node.left:
                if values[node.feature_i] < node.split_value:
                    node = node.left    
                else:
                    node = node.right
            predictions.append(node.predict_class)
        return np.array(predictions)
