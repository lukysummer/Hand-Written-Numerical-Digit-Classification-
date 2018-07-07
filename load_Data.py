# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:34:08 2018

@author: Luky
"""

'''
1. Load the data 
2. Plot the means for each of the digit classes.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def plot_means(train_data, train_labels):
    means = []
    
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)

        each_dig_1_64 = []
        each_dig_8_8 = []
        each_dig_1_64 = np.average(i_digits,axis=0)
        each_dig_1_64 = np.array(each_dig_1_64)
        
    
        for j in range(0,8):
            each_dig_8_8.append(each_dig_1_64[0+8*j:8+8*j])
            
        means.append(each_dig_8_8)
    
    #put the 10 8x8 images vertically side by side
    all_concat = np.concatenate(means, 1)
    
    
    # Plot all means of all 10 digits on same axis
    plt.imshow(all_concat, cmap='gray')
    plt.show()


if __name__ == '__main__':
    train_data, train_labels, _, _ = data.load_all_data_from_zip('a2digits.zip', 'data')
    plot_means(train_data, train_labels)