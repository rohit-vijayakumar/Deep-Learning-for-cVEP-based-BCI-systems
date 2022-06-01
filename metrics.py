import os
import glob
import copy
import time
import math
import pickle
import numpy as np

def calculate_ITR(num_classes, accuracy, time_min, num_trials):
    if(time_min==0):
        itr = 0
    else:  
        if (accuracy<0.1):
            B = 0
        elif(accuracy!=1):
            B = math.log(num_classes,2) + accuracy * math.log(accuracy,2) + (1-accuracy)*math.log(((1-accuracy)/(num_classes-1)),2)
        else:
            B = math.log(num_classes,2)
        
        Q = num_trials/time_min
        itr = B*Q
    return itr