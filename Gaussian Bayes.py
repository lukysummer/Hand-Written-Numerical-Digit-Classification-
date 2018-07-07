# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:41:26 2018

@author: Luky
"""

'''
Implement and evaluate the Conditional Gaussian classifier.
'''
from datetime import datetime
import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class:

    Return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''  
    means = []
    
    for k in range(0,10):
        means_each_digit = []
        
        for j in range(0,64):
            num = 0; denom = 0
            
            for i in range(0,len(train_data)):
                if(train_labels[i] == k):
                    num += train_data[i][j]
                    denom += 1

            means_each_digit.append(num/denom) 
        
        means.append(means_each_digit)

    means = np.array(means)
    
    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class:

    Return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    means = compute_mean_mles(train_data, train_labels)
    covariances = []
        
    for k in range(0,10):
        covariances_j = []
        
        for j in range(0,64):
            covariances_jj = []
                        
            for jj in range(0,64):
                num = 0; denom = 0
                
                for i in range(0,len(train_data)):
                    if(train_labels[i] == k):
                        num += (train_data[i][j] - means[k][j])*(train_data[i][jj] - means[k][jj])
                        denom += 1

                covariances_jj.append(num/denom) 
            
            covariances_j.append(covariances_jj)
        
        covariances.append(covariances_j)
    
    for k in range(10):
        for j in range(64):
            covariances[k][j][j] += 0.01
        
    covariances = np.array(covariances)    
    
    return covariances


def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    cov_diag = []
    for i in range(10):
        cov_diag.append(np.log(np.diag(covariances[i])))
    
    cov_diag_all_digits = []
    for i in range(10):
        img_i = cov_diag[i]
        
        cov_diag_8_8 = []
        for j in range(0,8):
           cov_diag_8_8.append(img_i[0+8*j:8+8*j])
            
        cov_diag_all_digits.append(cov_diag_8_8)
        
    all_concat = np.concatenate(cov_diag_all_digits, 1)
    
    plt.imshow(all_concat, cmap='gray')
    plt.show()


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)
    
        * digits: pixel values corresponding to each image. 
                ->  [N x 64] array where each row corresponds to a single data point.

    Should return an [n x 10] numpy array 
    '''  
    each_image = []
    
    for i in range(digits.shape[0]):  
        each_image_for_each_digit = [] 
        
        for k in range(10):
            two_pi_term = (np.log(2*np.pi))*(-0.5*digits.shape[1])
            det_covariance = (np.log(np.linalg.det(covariances[k])))*-0.5
            x_minus_mu = np.subtract(digits[i], means[k])
            inside_exponential = -0.5 * np.dot(np.dot(np.transpose(x_minus_mu),np.linalg.inv(covariances[k])), x_minus_mu)
            
            each_image_for_each_digit.append(two_pi_term + det_covariance + inside_exponential)
            
        each_image.append(each_image_for_each_digit)
    
    each_image = np.array(each_image)
    return each_image


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''  
    prior = [0.1]*len(digits) # [len(bin_digits) x 1]
    evidence = [] # [len(bin_digits) x 1]  *this term is SAME for all digits 0 to 9, since summed over k=0 to 9
    log_likelihood = generative_likelihood(digits, means, covariances) # [len(bin_digits) x 10]   
    cond_likelihood = [] # output # [len(bin_digits) x 10]
    
    # constructing evidence term, p(b|mean,covar): [n x 1] array
    for i in range(0,len(digits)):
        # this term is: p(y=k) * product over j=1 to d of p(b_j|y=k,mean_kj,covar_kj)
        each_image_evidence = 1 # scalar value
               
        for k in range(0,10):      
            two_pi_term = (2*np.pi)**(-0.5*digits.shape[1])
            det_covariance = (np.linalg.det(covariances[k]))**-0.5
            x_minus_mu = np.subtract(digits[i], means[k])
            inside_exponential = -0.5 * np.dot(np.dot(np.transpose(x_minus_mu),np.linalg.inv(covariances[k])), x_minus_mu)

            each_image_evidence = 0.1 * two_pi_term * det_covariance * np.exp(inside_exponential) # p(y=k) = 0.1
       
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


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances) 
    labels_sum = np.zeros(cond_likelihood.shape[0]) # sum of con_likelihood for true labels, to be averaged at the end
    
    for i in range(0,len(labels)): # from 0 to n-1
        k = int(labels[i])
        labels_sum[k] += cond_likelihood[i][k] # adding cond_likelihood for the correct class for a data point
   
    labels_avg = np.average(labels_sum) # scalar value

    return labels_avg


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances) # shape (n, 10)
    digit_prediction = []
    
    for j in range(0,len(cond_likelihood)):
        maxi = np.amax(cond_likelihood[j,:])
        
        for i in range(10):
            if(cond_likelihood[j][i]==maxi) :
                digit_prediction.append(i)
        
    return digit_prediction


def main():
    startTime = datetime.now()
    
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model on the training set
    means_train = compute_mean_mles(train_data, train_labels)
    covariances_train = compute_sigma_mles(train_data, train_labels)
    
    plot_cov_diagonal(covariances_train)
       
    avg_train = avg_conditional_likelihood(train_data, train_labels, means_train, covariances_train)
    print("")
    print("Average conditional log-likelihood of training set: ", avg_train)
    
    avg_test = avg_conditional_likelihood(test_data, test_labels, means_train, covariances_train)
    print("")
    print("Average conditional log-likelihood of test set: ", avg_test)
    print("")
    
    predictions_training = classify_data(train_data, means_train, covariances_train) # (n_train x 1) matrix
    predictions_test = classify_data(test_data, means_train, covariances_train) # (n_test x 1) matrix
    
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
    print("F1 score for training set: ", F1_training)
    print("")
    print("training precision & recall: ", precision_training, "& ", recall_training)
    
    F1_test = (2 * precision_test * recall_test)/(precision_test + recall_test)
    print("F1 score for test set: ", F1_test)
    print("")
    print("test precision & recall: ", precision_test, "& ", recall_test)
    
    
    correct_train = 0
    correct_test = 0
    for i in range(0,len(predictions_training)):
        if(predictions_training[i] == train_labels[i]):
            correct_train += 1
    for i in range(0,len(predictions_test)):
        if(predictions_test[i] == test_labels[i]):
            correct_test += 1


    lead_vec = []
    for i in range(len(covariances_train)):
        w, v = np.linalg.eig(covariances_train[i])
        lead_val = np.amax(w) # largest eigenvalue
        lead_index = [i for i, x in enumerate(w) if x == lead_val]
        lead_vec.append((v[:,lead_index].T)[0])
    
        
    eig_all_digits = []
    for i in range(10):
        img_i = lead_vec[i]
        
        eig_8_8 = []
        for j in range(0,8):
           eig_8_8.append(img_i[0+8*j:8+8*j])
            
        eig_all_digits.append(eig_8_8)
        
    all_concat = np.concatenate(eig_all_digits, 1)
    
    plt.imshow(all_concat, cmap='gray')
    plt.show()
    
    print(""); print("")
    print("Runtime: ", datetime.now() - startTime)

if __name__ == '__main__':
    main()