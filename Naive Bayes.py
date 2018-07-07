# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:42:23 2018

@author: Luky
"""

'''
Implementation & Evaluation of Naive Bayes classifier.
'''
from datetime import datetime
import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def binarize_data(pixel_values):
    '''
    1. Binarize the data by thresholding around 0.5
    '''
    #  in main: train_data, test_data = binarize_data(train_data), binarize_data(test_data) 
    return np.where(pixel_values > 0.5, 1.0, 0.0)


def compute_parameters(train_data, train_labels):
    '''
    2. Compute the eta MAP estimate/MLE with augmented data

    Return a numpy array of shape (10, 64), where the ith row corresponds to the ith digit class.

    '''    
    # insert 2 additional points in training data for each digit
    for i in range(0,10):
        train_data = np.vstack([train_data, np.ones(64)])
        train_data = np.vstack([train_data, np.zeros(64)])
    
    # insert 2 additional labels in training labels for each digit    
    for i in range(0,10):
        train_labels = np.insert(train_labels,len(train_labels),i)
        train_labels = np.insert(train_labels,len(train_labels),i)
     
    # shuffle new train data with 2 additional points
    train_indices = np.random.permutation(train_data.shape[0])
    train_data, train_labels = train_data[train_indices], train_labels[train_indices]
    
    '''
    Now with 2 additional training points for each digit,
    train_data : [7020,64] -> 700 test points for each of 10 digits, randomly shuffled
    train_labels : [7020,1] -> 700 labels for each of 10 digits, randomly shuffled
    '''    
    eta = []
    
    for k in range(0,10):
        eta_each_digit = []
        
        for j in range(0,64):
            num = 0; denom = 0
            
            for i in range(0,len(train_data)):
                if(train_labels[i] == k):
                    num += train_data[i][j]
                    denom += 1

            eta_each_digit.append(num/denom) 
        
        eta.append(eta_each_digit)
    
    eta = np.array(eta)
    return eta


def plot_images(class_images):
    '''
    3. Plot each of the images corresponding to each class side by side in grayscale
    '''
    eta_all_digits = []
    
    for i in range(10):
        img_i = class_images[i] # 1 x 64
        
        eta_8_8 = []
        for j in range(0,8):
           eta_8_8.append(img_i[0+8*j:8+8*j]) # 8 x 8
            
        eta_all_digits.append(eta_8_8)
        
    all_concat = np.concatenate(eta_all_digits, 1)
    
    plt.imshow(all_concat, cmap='gray')
    plt.show()




def generate_new_data(eta):
    '''
    4. Sample a new data point from your generative distribution p(x|y,theta) for each value of y in range 0...10
       & Plot these values
    '''
    generated_data = []
    
    for k in range(0,10):
        each_dig = []
        
        for j in range(0,64):
            if(eta[k][j] > 0.5):
                each_dig.append(1)
            else:
                each_dig.append(0)          
    
        generated_data.append(each_dig)
    
    plot_images(generated_data)


def generative_likelihood(bin_digits, eta):
    '''
    5. Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an [n x 10] numpy array, -> 
    Where n is the number of datapoints and 10 corresponds to each digit class
        
    "bin_digits" : pixel values corresponding to each image. 
    = Nx64 array where each row corresponds to a single data point.

    train_data : [7000,64] -> 700 test points for each of 10 digits
    test_data : [4000,64] -> 400 test points for each of 10 digits
    '''
    each_digit = []

    for i in range(0,len(bin_digits)):
        each_image = [] 
        
        for k in range(0,10):
            each_pixel = 0
            
            for j in range(0,64):
                each_pixel += bin_digits[i][j]*np.log(eta[k][j]) + (1-bin_digits[i][j])*np.log(1-eta[k][j])
     
            each_image.append(each_pixel) # each_image i
   
        each_digit.append(each_image)
    
    each_digit = np.array(each_digit)
        
    return each_digit


def conditional_likelihood(bin_digits, eta):
    '''
    5. Compute the conditional log likelihood:   log p(y|x, eta)
    This should be a numpy array of shape (n, 10), where n is the number of datapoints and 10 corresponds to each digit class
    '''
    prior = [0.1]*len(bin_digits) # [len(bin_digits) x 1]
    evidence = [] # [len(bin_digits) x 1]  *this term is SAME for all digits 0 to 9, since summed over k=0 to 9
    log_likelihood = generative_likelihood(bin_digits,eta) # [len(bin_digits) x 10]
    cond_likelihood = [] # output # [len(bin_digits) x 10]
    
    # constructing evidence term, p(b|eta): [n x 1] array
    for i in range(0,len(bin_digits)):
        each_image_evidence = 0 # scalar value
               
        for k in range(0,10):
            # this subterm is: product over j=1 to d of p(b_j|y=k,eta_kj)
            each_image_evidence_subterm = 1 # scalar value
            
            for j in range(0,64):
                each_image_evidence_subterm *= ((eta[k][j])**bin_digits[i][j])*((1-eta[k][j])**(1-bin_digits[i][j]))

            each_image_evidence += 0.1*each_image_evidence_subterm # p(y=k) = 0.1
       
        evidence.append(each_image_evidence)
    
    
    # constructing conditional likelihood
    for k in range(0,10):
        each_digit_term = []
        each_digit_term = np.subtract(np.add(np.log(prior),log_likelihood[:,k]),np.log(evidence))
        each_digit_term = np.array([each_digit_term])
        each_digit_term = np.transpose(each_digit_term)

        if(k == 0):
            cond_likelihood = each_digit_term
            cond_likelihood = np.array(cond_likelihood)
        else:
            cond_likelihood = np.append(cond_likelihood, each_digit_term, axis=1)
    
    cond_likelihood = np.array(cond_likelihood)
    
    return cond_likelihood


def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    5. Compute the (scalar) average conditional log likelihood over the TRUE (=CORRECT) class labels ONLY

        AVG( log p(y_i|x_i, eta) )

        i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)  # n x 10
    labels_sum = np.zeros(cond_likelihood.shape[0]) # sum of con_likelihood for true labels, to be averaged at the end
    
    # len(lables) = cond_likelihood.shape[0] = n where n is # of data point
    for i in range(0,len(labels)): # from 0 to n-1
        k = int(labels[i])
        labels_sum[k] += cond_likelihood[i][k] # adding cond_likelihood for the correct class for a data point
   
    labels_avg = np.average(labels_sum) # scalar value

    return labels_avg


def classify_data(bin_digits, eta):
    '''
    6. Classify new points by taking the most likely posterior class
       i.e. compute and return the most likely class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    digit_prediction = []
    
    for j in range(0,len(cond_likelihood)):
        maxi = np.amax(cond_likelihood[j,:])
        
        for i in range(10):
            if(cond_likelihood[j][i]==maxi) :
                digit_prediction.append(i)
        
    return digit_prediction


def main():
    startTime = datetime.now()
    
    # Load data
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)
    
    '''
    train_data : [7000,64] -> 700 test points for each of 10 digits, randomly shuffled
    train_labels : [7000,1] -> 700 labels for each of 10 digits, randomly shuffled
    test_data : [4000,64] -> 400 test points for each of 10 digits, randomly shuffled
    test_labels : [4000,1] -> 400 labels for each 400 test points for each of 10 digits, randomly shuffled
    '''
        
    # Fit the model on the training set
    eta_train = compute_parameters(train_data, train_labels)

    plot_images(eta_train)

    generate_new_data(eta_train)
    
    print("")
    avg_train = avg_conditional_likelihood(train_data, train_labels, eta_train)
    print("The average conditional log-likelihood of the train set : ", avg_train)
    
    print("")
    avg_test = avg_conditional_likelihood(test_data, test_labels, eta_train)
    print("The average conditional log-likelihood of the test set : ", avg_test)
 
    predictions_training = classify_data(train_data, eta_train) # (n_train x 1) matrix
    predictions_test = classify_data(test_data, eta_train) # (n_test x 1) matrix
    
    train_accuracy_matrix = np.zeros((10,10))
    test_accuracy_matrix = np.zeros((10,10))
    
    for i in range(0,len(predictions_training)): 
        k = int(train_labels[i])    
        train_accuracy_matrix[k][predictions_training[i]] += 1
    
    for i in range(0,len(predictions_test)): 
        k = int(test_labels[i])   
        test_accuracy_matrix[k][predictions_test[i]] += 1         

    '''  TP : True Positive | FN :  False Negative | FP : True Positive  '''
    
    # Calculate Recall = (TP)/(TP+FN)
    recall_training_for_each_true_label = []
    recall_test_for_each_true_label = []
    
    for i in range(0,10): # iterate for each true label
        TP_plus_FN_training = 0
        TP_plus_FN_test = 0    
        
        for j in range(0,10): # iterate for each predicted label
            TP_plus_FN_training += train_accuracy_matrix[i][j]
            TP_plus_FN_test += test_accuracy_matrix[i][j]
        
        # TP = train/test_accuracy_matrix[i][i]      
        recall_training_for_each_true_label.append((train_accuracy_matrix[i][i])/TP_plus_FN_training) # (TP)/(TP+FN)
        recall_test_for_each_true_label.append((test_accuracy_matrix[i][i])/TP_plus_FN_test)
   
    # averaging over all true labels          
    recall_training = np.average(recall_training_for_each_true_label)
    recall_test = np.average(recall_test_for_each_true_label)
    
    # Calculate Precision = (TP)/(TP+FP)
    precision_training_for_each_predicted_label = []
    precision_test_for_each_predicted_label = []
    
    for j in range(0,10): # iterate for each true label
        TP_plus_FP_training = 0
        TP_plus_FP_test = 0
        
        for i in range(0,10): # iterate for each predicted label
            TP_plus_FP_training += train_accuracy_matrix[i][j]
            TP_plus_FP_test += test_accuracy_matrix[i][j]
        
        precision_training_for_each_predicted_label.append((train_accuracy_matrix[j][j])/TP_plus_FP_training) # (TP)/(TP+FP)
        precision_test_for_each_predicted_label.append((test_accuracy_matrix[j][j])/TP_plus_FP_test)
   
    # averaging over all predicted labels    
    precision_training = np.average(precision_training_for_each_predicted_label)
    precision_test = np.average(precision_test_for_each_predicted_label)
    
    # Calculate F1 Score = (2*Precision*Recall)/(Precision + Recall)
    F1_training = (2 * precision_training * recall_training)/(precision_training + recall_training)
    F1_test = (2 * precision_test * recall_test)/(precision_test + recall_test)

    print("")
    print("training precision & recall: ", precision_training, "& ", recall_training)
    print("test precision & recall: ", precision_test, "& ", recall_test)
    print("")
    print("F1 score for training set : ", F1_training)
    print("F1 score for test set : ", F1_test)
    
    correct_train = 0
    correct_test = 0
    
    for i in range(0,len(predictions_training)):
        if(predictions_training[i] == train_labels[i]):
            correct_train += 1
            
    for i in range(0,len(predictions_test)):
        if(predictions_test[i] == test_labels[i]):
            correct_test += 1
    
 
    print("");print("")
    print("Runtime: ", datetime.now() - startTime)
    
if __name__ == '__main__':
    main()