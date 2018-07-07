# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:39:59 2018

@author: Luky
"""

'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''
from datetime import datetime
import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels


    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)
    

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm
        
        Return the digit label provided by the algorithm
        '''
        distances = KNearestNeighbor.l2_distance(self,test_point)
        k_nearest_labels = [] # gonna be [k x 1] array
        
        # finding k smallest distances 
        for j in range(0,k):
            mini = np.amin(distances)
            mini_index = [i for i, x in enumerate(distances) if x == mini]
            k_nearest_labels.append(self.train_labels[mini_index[0]])
            
            distances = np.delete(distances,mini_index[0])
        # Now, the elements of "k_nearest_labels" array are k labels of k-smallest-distanced training points.
        
        digit, cnt = np.unique(k_nearest_labels, return_counts=True)
        frequency = np.asarray((digit, cnt))   # array listing how many times each label appears in k_nearest_labels
        # frequency[0] lists all labels that appeared in k_nearest_labels
        # frequency[1] lists how many times each label appeared in k_nearest_labels
        
        most_frequent = np.amax(frequency[1]) # number of appearances of the most frequent label(s)
        most_frequent_index = [i for i, x in enumerate(frequency[1]) if x == most_frequent]
        
        
        if(len(most_frequent_index) == 1): # only one label with most appearances (unique majority label) -> no tiebreaker required
            prediction = frequency[0][most_frequent_index[0]]
            
        else: # TIEBREAKER
            found = False
            for i in range(len(k_nearest_labels)):    
                for j in range((len(most_frequent_index))):
                    if(k_nearest_labels[i] == frequency[0][most_frequent_index[j]]):
                        prediction = k_nearest_labels[i] 
                        found = True
                        break
                if found:
                    break
        
        return prediction


def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
        
        kf = KFold(n_splits=10)
        F1_sum = [] #sum over 10 folds
        
        fold = 1
        for train_index, valid_index in kf.split(train_data):            
            # Splitting training data into training & validation sets
            dig_train, dig_valid = train_data[train_index], train_data[valid_index]
            label_train, label_valid = train_labels[train_index], train_labels[valid_index]
            
            knn = KNearestNeighbor(dig_train, label_train)
            
            for k in k_range:
                
                predictions = []
                
                for i in range(np.array(dig_valid).shape[0]):
                    predictions.append(knn.query_knn(dig_valid[i], k))
                
                F1_score = classification_accuracy(knn, k, predictions, label_valid)
                
                if(fold==1):
                    F1_sum.append(F1_score)
                
                else:
                    F1_sum[k-1] += F1_score
             
            fold += 1
                    

        F1_avg = np.divide(F1_sum,10) #averaging over 10 folds
        for i in range(len(F1_avg)):
            K = i+1
            print("F1 Score for k=", K, ": ", F1_avg[i])

        max_F1 = np.amax(F1_avg)
        max_F1_index = [i for i, x in enumerate(F1_avg) if x == max_F1]
        
        avg_F1 = np.average(F1_avg)
        optimal_K = max_F1_index[0] + 1
        
        return optimal_K, max_F1, avg_F1
    
    

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data', using eval_labels
    '''
    accuracy_matrix = np.zeros((10,10))
    
    for i in range(0,len(eval_data)): 
        accuracy_matrix[int(eval_labels[i])][int(eval_data[i])] += 1               

    '''  TP : True Positive | FN :  False Negative | FP : True Positive  '''    
    # Calculate Recall = (TP)/(TP+FN)
    recall_for_each_true_label = []
    for i in range(0,10): # iterate for each true label
        TP_plus_FN = 0
  
        for j in range(0,10): # iterate for each predicted label
            TP_plus_FN += accuracy_matrix[i][j]

        recall_for_each_true_label.append((accuracy_matrix[i][i])/TP_plus_FN) # (TP)/(TP+FN)
 
    # averaging over all true labels    
    recall = np.average(recall_for_each_true_label)

    
    # Calculate Precision = (TP)/(TP+FP)
    precision_for_each_predicted_label = []
    for j in range(0,10): # iterate for each true label
        TP_plus_FP = 0

        for i in range(0,10): # iterate for each predicted label
            TP_plus_FP += accuracy_matrix[i][j]

        precision_for_each_predicted_label.append((accuracy_matrix[j][j])/TP_plus_FP) # (TP)/(TP+FP)

    # averaging over all predicted labels    
    precision = np.average(precision_for_each_predicted_label)
    
    # Calculate F1 Score = (2*Precision*Recall)/(Precision + Recall)
    F1_Score = (2 * precision * recall)/(precision + recall)

    return F1_Score


def main():
    startTime = datetime.now()
    
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)
    '''
    train_data : [7000,64] -> 700 test points for each of 10 digits, randomly shuffled
    train_labels : [7000,1] -> 700 labels for each of 10 digits, randomly shuffled
    test_data : [4000,64] -> 400 test points for each of 10 digits, randomly shuffled
    test_labels : [4000,1] -> 400 labels for each 400 test points for each of 10 digits, randomly shuffled
    '''
    
    # 1-NN
    count_train_1 = 0
    predictions_train_k_1 = []
    
    for i in range(np.array(train_data).shape[0]):
        predictions_train_k_1.append(knn.query_knn(train_data[i], 12))
        if(predictions_train_k_1[i] == train_labels[i]):
            count_train_1 += 1
    
    F1_train_k_1 = classification_accuracy(knn, 1, predictions_train_k_1, train_labels)
    print("< k=1 >")
    print("")
    print("train F1 score: ", F1_train_k_1)
    print("Number of correct predictions for k=1 for training stage : ", count_train_1)
    
    count_test_1 = 0        
    predictions_test_k_1 = []
    
    for i in range(np.array(test_data).shape[0]):
        predictions_test_k_1.append(knn.query_knn(test_data[i], 1))
        if(predictions_test_k_1[i] == test_labels[i]):
            count_test_1 += 1

    F1_test_k_1 = classification_accuracy(knn, 1, predictions_test_k_1, test_labels)
        
    print("")
    print("test F1 score: ", F1_test_k_1)
    print("Number of correct predictions for k=1 for test stage : ", count_test_1)
    
    
    # 15-NN
    count_train_15 = 0
    predictions_train_k_15 = []
    
    for i in range(np.array(train_data).shape[0]):
        predictions_train_k_15.append(knn.query_knn(train_data[i], 15))
        if(predictions_train_k_15[i] == train_labels[i]):
            count_train_15 += 1
    
    F1_train_k_15 = classification_accuracy(knn, 15, predictions_train_k_15, train_labels)
    print("")
    print("< k=15 >")
    print("")
    print("train F1 score: ", F1_train_k_15)
    print("Number of correct predictions for k=15 for training stage: ", count_train_15)
    print("")
    
    count_test_15 = 0        
    predictions_test_k_15 = []
    
    for i in range(np.array(test_data).shape[0]):
        predictions_test_k_15.append(knn.query_knn(test_data[i], 15))
        if(predictions_test_k_15[i] == test_labels[i]):
            count_test_15 += 1

    F1_test_k_15 = classification_accuracy(knn, 15, predictions_test_k_15, test_labels)
        
    print("")
    print("test F1 score: ", F1_test_k_15)
    print("Number of correct predictions for k=15 for test stage: ", count_test_15)
    print("")
    
    # k-fold cross validation   
    optimal_K, max_F1, avg_F1 = cross_validation(train_data, train_labels)
    print("")
    print("The Optimal K for training set is: ", optimal_K)
    print("")
    print("The train classification accuracy with this Optimal K is: ", max_F1)
    print("")
    print("The average accuracy across folds is: ", avg_F1)
        
    predictions_test = []
    for i in range(np.array(test_data).shape[0]):
        predictions_test.append(knn.query_knn(test_data[i], optimal_K))

    F1_test = classification_accuracy(knn, optimal_K, predictions_test, test_labels)
        
    print("")
    print("test F1 score with this Optimal K is: ", F1_test)
    
    print("")
    print("Runtime: ", datetime.now() - startTime)   


if __name__ == '__main__':
    main()