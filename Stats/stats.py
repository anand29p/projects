# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 09:27:25 2021

@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np

education = pd.read_csv(r"C:\Users\USER\Documents\education.csv")

education.describe()
education.workex.mean()
education.workex.median()
education.workex.mode()

education.gmat.mean()
education.gmat.median()
education.gmat.mode()

from scipy import stats
stats.mode(education.workex)

# Measure of dispersion / second moment business decision
education.workex.var() # Variance
education.workex.std() # Standard Deviation
range= max(education.workex) - min(education.workex)
range

# Plots
plt.bar(height = education.gmat, x = np.arange(1,774,1)) # initializing the parameter

plt.hist(education.gmat) #histogram
plt.hist(education.workex) #histogram

#Third moment business decision
education.gmat.skew()
education.workex.skew()

#Fourth moment business decision
education.gmat.kurt()
education.workex.kurt()

plt.boxplot(education.gmat) #boxplot
plt.boxplot(education.workex) #boxplot