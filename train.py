import os
import glob
import copy
import time
import math
import pickle
import numpy as np
import scipy.io
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from data_preprocessing import*
from metrics import *

from sklearn.cross_decomposition import CCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2,l1
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

dataset = '8_channel_cVEP'
mode = 'within_subject'
model = 'cca

if(model=='cca'):
    run_cca(dataset, mode)
