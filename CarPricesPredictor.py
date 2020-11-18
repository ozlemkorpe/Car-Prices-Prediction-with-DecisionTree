#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import stats
from scipy.stats import norm, skew

#Preprocessing

#Read dataset
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

data_train.head().T