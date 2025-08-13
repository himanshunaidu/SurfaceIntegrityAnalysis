import numpy as np
import pandas as pd
import scipy.stats as stats

arr1 = np.arange(50, 60, 1)
arr1 = np.concatenate((arr1, np.arange(50, 60, 1)))
arr1 = np.concatenate((arr1, np.arange(50, 60, 1)))
arr1 = np.concatenate((arr1, np.arange(50, 60, 1)))
print(arr1)

mean = np.mean(arr1)
std = np.std(arr1)
kurtosis = stats.kurtosis(arr1, fisher=False)
print("Mean", mean, "Standard Deviation:", std, "Kurtosis:", kurtosis)

arr2 = np.ones(20) * 0
arr2 = np.concatenate((arr2, np.ones(20) * 100))
print(arr2)

mean = np.mean(arr2)
std = np.std(arr2)
kurtosis = stats.kurtosis(arr2, fisher=False)
print("Mean", mean, "Standard Deviation:", std, "Kurtosis:", kurtosis)
