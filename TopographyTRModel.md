This notebook is for the WIP paper tentatively titled Response Topography Modulates Response-Withholding Performance in Mice by Daniels et al (in prep).  
*Note: A draft of this paper is not yet available.*   
*Note: Some of this code is back from when I first transitioned from Matlab to Python; thus best practices are not always adhered too; I apologize in advance*

The goal of this study was to **determine the ideal parameters for training mice in a response withholding task under varying levels of motivation**. 
1. To that end, mice were divided into two different groups: initiation/first response as nose-poke (NP) or  lever-press. (LP) 
2. Both groups of mice were then trained in sequence:  differential-reinforcement-of-low rates (DRL) and then a fixed-minimum interval (FMI) schedule of reinforcement. 
3. In each schedule two intervals were explored: 3 s and 6 s; along with two levels of food deprivation. 
4. Once training stabilized in each interval mice were given a pre-feeding challenge (1 hour of food access immediately prior to the experimental session). 

The following data were collected:
1. Latencies: the time to the first response 
2. Inter-response interval (IRT): the time between the first response and the second


Data were originally processed in Matlab (for historical reasons). Rather than repeat such processing, the .mat files and .m files can be found in the Github for this project. The remainder of the processing and analysis needed will be conducted in this Jupyter Notebook unless otherwise specified. 

Specifically,
1. Latencies will be analyzed via an empirical analysis. 
2. IRTs will be analyzed by fitting the Temporal Regulation model (TR: see Watterson et al., 2015) and an extended version of the TR model, which I refer to as TRe. 

Briefly, the TR model assumes that mice fluctuate in and out of two engagement states and that there is a timing state whose IRTs can be described by a gamma distribution and a non-timing state whose IRTs can be described by an exponential distribution. Alternatively, the TRe model assumes that mice fluctuate in and out of three engagement states: the timing state, a burst responding state, and a lapse state. Whereas the timing state can be described as before; the burst responding and lapse states are both described by exponential distributions but with different means. The mean of the burst responding state must be shorter than the mean of the lapse responding state. 

Without belaboring any points or giving away any details crucial for the analysis, let's pull in the packages we will need. 


```python
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy.stats import gamma
from scipy.stats import expon
import os
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg
import re
```

First we will import the database in storing all IRTs from the experiment and parameter estimates following TR model fitting.


```python
location = "E:\\Dropbox\\2 All of Carter's Stuff\\Carter Local\\ASU Research Projects\\Mutant Mice Projects\\Topography Project\\New Analysis (based on Watterson et al. 2015)\\IRTs_TRmodel1.mat"
matlab_database = spio.loadmat(location, squeeze_me = True)
matlab_names = spio.whosmat(location)
for names in matlab_names:
    print(names)
```

    ('A', (1, 1), 'double')
    ('DRL_exp1', (3828, 19), 'double')
    ('DRL_exp1_PF', (493, 19), 'double')
    ('DRL_exp2', (6211, 19), 'double')
    ('DRL_exp2_PF', (828, 19), 'double')
    ('FI', (1, 1), 'double')
    ('FMI_exp1', (1699, 19), 'double')
    ('FMI_exp1_PF', (276, 19), 'double')
    ('FMI_exp2', (2178, 19), 'double')
    ('FMI_exp2_PF', (284, 19), 'double')
    ('IRTs', (3215, 1), 'double')
    ('Intervals', (2178, 1), 'double')
    ('LL_out', (1, 1), 'double')
    ('LLcollector', (1, 16), 'double')
    ('LLcollector_DRL_e1_B3', (1, 16), 'double')
    ('LLcollector_DRL_e1_B3_TRe', (1, 16), 'double')
    ('LLcollector_DRL_e1_B6', (1, 16), 'double')
    ('LLcollector_DRL_e1_B6_TRe', (1, 16), 'double')
    ('LLcollector_DRL_e1_PF3', (1, 16), 'double')
    ('LLcollector_DRL_e1_PF3_TRe', (1, 16), 'double')
    ('LLcollector_DRL_e1_PF6', (1, 16), 'double')
    ('LLcollector_DRL_e1_PF6_TRe', (1, 16), 'double')
    ('LLcollector_DRL_e2_B3', (1, 16), 'double')
    ('LLcollector_DRL_e2_B3_TRe', (1, 16), 'double')
    ('LLcollector_DRL_e2_B6', (1, 16), 'double')
    ('LLcollector_DRL_e2_B6_TRe', (1, 16), 'double')
    ('LLcollector_DRL_e2_PF3', (1, 16), 'double')
    ('LLcollector_DRL_e2_PF3_TRe', (1, 16), 'double')
    ('LLcollector_DRL_e2_PF6', (1, 16), 'double')
    ('LLcollector_DRL_e2_PF6_TRe', (1, 16), 'double')
    ('LLcollector_FMI_e1_B3', (1, 16), 'double')
    ('LLcollector_FMI_e1_B3_TRe', (1, 16), 'double')
    ('LLcollector_FMI_e1_B6', (1, 16), 'double')
    ('LLcollector_FMI_e1_B6_TRe', (1, 16), 'double')
    ('LLcollector_FMI_e1_PF3', (1, 16), 'double')
    ('LLcollector_FMI_e1_PF3_TRe', (1, 16), 'double')
    ('LLcollector_FMI_e1_PF6', (1, 16), 'double')
    ('LLcollector_FMI_e1_PF6_TRe', (1, 16), 'double')
    ('LLcollector_FMI_e2_B3', (1, 16), 'double')
    ('LLcollector_FMI_e2_B3_TRe', (1, 16), 'double')
    ('LLcollector_FMI_e2_B6', (1, 16), 'double')
    ('LLcollector_FMI_e2_B6_TRe', (1, 16), 'double')
    ('LLcollector_FMI_e2_PF3', (1, 16), 'double')
    ('LLcollector_FMI_e2_PF3_TRe', (1, 16), 'double')
    ('LLcollector_FMI_e2_PF6', (1, 16), 'double')
    ('LLcollector_FMI_e2_PF6_TRe', (1, 16), 'double')
    ('O', (1, 31), 'double')
    ('Subjects', (1, 1), 'double')
    ('Wait3data', (789, 2), 'double')
    ('Wait6data', (686, 2), 'double')
    ('a', (789, 1), 'double')
    ('a2', (686, 1), 'double')
    ('ans', (1, 1), 'double')
    ('b', (789, 1), 'double')
    ('b2', (686, 1), 'double')
    ('data', (2470, 1), 'double')
    ('data_aspdf', (1, 30), 'double')
    ('edges', (1, 31), 'double')
    ('exitFlag', (1, 1), 'double')
    ('i', (1, 1), 'double')
    ('mean', (1, 1), 'double')
    ('p_fit', (6, 1), 'double')
    ('param_init', (1, 6), 'double')
    ('param_lb', (1, 8), 'double')
    ('param_ub', (1, 6), 'double')
    ('pcollector', (6, 16), 'double')
    ('pcollector_DRL_e1_B3', (4, 16), 'double')
    ('pcollector_DRL_e1_B3_TRe', (6, 16), 'double')
    ('pcollector_DRL_e1_B6', (4, 16), 'double')
    ('pcollector_DRL_e1_B6_TRe', (6, 16), 'double')
    ('pcollector_DRL_e1_PF3', (4, 16), 'double')
    ('pcollector_DRL_e1_PF3_TRe', (6, 16), 'double')
    ('pcollector_DRL_e1_PF6', (4, 16), 'double')
    ('pcollector_DRL_e1_PF6_TRe', (6, 16), 'double')
    ('pcollector_DRL_e2_B3', (4, 16), 'double')
    ('pcollector_DRL_e2_B3_TRe', (6, 16), 'double')
    ('pcollector_DRL_e2_B6', (4, 16), 'double')
    ('pcollector_DRL_e2_B6_TRe', (6, 16), 'double')
    ('pcollector_DRL_e2_PF3', (4, 16), 'double')
    ('pcollector_DRL_e2_PF3_TRe', (6, 16), 'double')
    ('pcollector_DRL_e2_PF6', (4, 16), 'double')
    ('pcollector_DRL_e2_PF6_TRe', (6, 16), 'double')
    ('pcollector_FMI_e1_B3', (4, 16), 'double')
    ('pcollector_FMI_e1_B3_TRe', (6, 16), 'double')
    ('pcollector_FMI_e1_B6', (4, 16), 'double')
    ('pcollector_FMI_e1_B6_TRe', (6, 16), 'double')
    ('pcollector_FMI_e1_PF3', (4, 16), 'double')
    ('pcollector_FMI_e1_PF3_TRe', (6, 16), 'double')
    ('pcollector_FMI_e1_PF6', (4, 16), 'double')
    ('pcollector_FMI_e1_PF6_TRe', (6, 16), 'double')
    ('pcollector_FMI_e2_B3', (4, 16), 'double')
    ('pcollector_FMI_e2_B3_TRe', (6, 16), 'double')
    ('pcollector_FMI_e2_B6', (4, 16), 'double')
    ('pcollector_FMI_e2_B6_TRe', (6, 16), 'double')
    ('pcollector_FMI_e2_PF3', (4, 16), 'double')
    ('pcollector_FMI_e2_PF3_TRe', (6, 16), 'double')
    ('pcollector_FMI_e2_PF6', (4, 16), 'double')
    ('pcollector_FMI_e2_PF6_TRe', (6, 16), 'double')
    ('sd', (1, 1), 'double')
    ('visualize', (434, 1), 'double')
    ('visualize3', (282, 1), 'double')
    ('visualize6', (417, 1), 'double')
    ('visualize_pdf', (1, 30), 'double')
    ('visualizecdf_bins', (1, 30), 'double')
    ('x', (434, 1), 'double')
    ('x3', (282, 1), 'double')
    ('x6', (417, 1), 'double')
    ('y', (434, 1), 'double')
    ('y3', (282, 1), 'double')
    ('y6', (417, 1), 'double')
    

As you can see, this matlab database contains a lot of information. For our first analysis, which is a visualization of model fit we need (1) IRTs for each condition and (2) parameter estimates in each condition. 

IRTs can be found in variables titled Schedule_expX?_PF where Schedule = DRL or FMI X = 1 or 2 and the baseline condition has nothing else but the pre-feeding conditions has _PF.

Parameter estimates can be found in pcollector_Schedule_expX where Schedule and X are as in above for the baseline condition and parameter estimates for the pre-feeding condition are as follows pcollector_Schedule_expX?_PF.

Let's focusing on constructing obtained probability functions. We will start with DRL and move our away through the waiting requirements,schedules, and feeding conditions until we finish with FMI 6-s.

Note that because we will do some model selection we are going to gather counts of the amount of data. 


```python
##DRL Data. Note that due to a chamber malfunction subject 1 does not have reliable pre-feeding for 6-s
#Baseline
data_counts = np.zeros([2,16])
E1_DRL_B = matlab_database['DRL_exp1'] #Note that the last column indicates the waiting requirement
E1_DRL_B3 = E1_DRL_B[E1_DRL_B[:,-1]==3,2:-1] #Select data only for 3 s condition, dropping the waiting requirement
databins = np.linspace(0,15,31)
E1_DRL_B_3hist = np.zeros((databins.shape[0]-1,16)) 
for i in range(0,16): 
    data = E1_DRL_B3[:,i][~np.isnan(E1_DRL_B3[:,i])]
    data_counts[0,i] = data_counts[0,i] + np.count_nonzero(E1_DRL_B3[:,i][~np.isnan(E1_DRL_B3[:,i])])
    output = np.histogram(data,bins=databins, density=False)
    E1_DRL_B_3hist[:,i] = np.divide(output[:][0],np.sum(output[:][0]))
    
E1_DRL_B6 = E1_DRL_B[E1_DRL_B[:,-1]==6,2:-1] #Select data only for 6 s condition, dropping needless columns
E1_DRL_B_6hist = np.zeros((databins.shape[0]-1,16)) #Empty array
for i in range(0,15):
    data = E1_DRL_B6[:,i][~np.isnan(E1_DRL_B6[:,i])]
    data_counts[0,i] = data_counts[0,i] + np.count_nonzero(E1_DRL_B6[:,i][~np.isnan(E1_DRL_B6[:,i])])
    output = np.histogram(data,bins=databins, density=False)
    E1_DRL_B_6hist[:,i] = np.divide(output[:][0],np.sum(output[:][0]))
    
#Pre-feeding
E1_DRL_PF = matlab_database['DRL_exp1_PF'] #Note that the last column indicates the waiting requirement
E1_DRL_PF3 = E1_DRL_PF[E1_DRL_PF[:,-1]==3,2:-1] #Select data only for 3 s condition, dropping the waiting requirement
E1_DRL_PF_3hist = np.zeros((databins.shape[0]-1,16)) 
for i in range(0,16): 
    data = E1_DRL_PF3[:,i][~np.isnan(E1_DRL_PF3[:,i])]
    data_counts[0,i] = data_counts[0,i] + np.count_nonzero(E1_DRL_PF3[:,i][~np.isnan(E1_DRL_PF3[:,i])])
    output = np.histogram(data,bins=databins, density=False)
    E1_DRL_PF_3hist[:,i] = np.divide(output[:][0],np.sum(output[:][0]))
    
E1_DRL_PF6 = E1_DRL_PF[E1_DRL_PF[:,-1]==6,2:-1] #Select data only for 6 s condition, dropping needless columns
E1_DRL_PF_6hist = np.zeros((databins.shape[0]-1,16)) #Empty array
for i in range(0,15):
    data = E1_DRL_PF6[:,i][~np.isnan(E1_DRL_PF6[:,i])]
    data_counts[0,i] = data_counts[0,i] + np.count_nonzero(E1_DRL_PF6[:,i][~np.isnan(E1_DRL_PF6[:,i])])
    output = np.histogram(data,bins=databins, density=False)
    E1_DRL_PF_6hist[:,i] = np.divide(output[:][0],np.sum(output[:][0]))
    
#FMI Data. Note that Subject 16 died during the FMI condition and thus has no data
#Baseline
E1_FMI_B = matlab_database['FMI_exp1'] #Note that the last column indicates the waiting requirement
E1_FMI_B3 = E1_FMI_B[E1_FMI_B[:,-1]==3,2:-1] #Select data only for 3 s condition, dropping needless columns
E1_FMI_B_3hist = np.zeros((databins.shape[0]-1,15)) 
for i in range(0,15): 
    data = E1_FMI_B3[:,i][~np.isnan(E1_FMI_B3[:,i])]
    data_counts[1,i] = data_counts[1,i] + np.count_nonzero(E1_FMI_B3[:,i][~np.isnan(E1_FMI_B3[:,i])])
    output = np.histogram(data,bins=databins, density=False)
    E1_FMI_B_3hist[:,i] = np.divide(output[:][0],np.sum(output[:][0]))
    
E1_FMI_B6 = E1_FMI_B[E1_FMI_B[:,-1]==6,2:-1] #Select data only for 6 s condition, dropping needless columns
E1_FMI_B_6hist = np.zeros((databins.shape[0]-1,15)) #Empty array
for i in range(0,15):
    data = E1_FMI_B6[:,i][~np.isnan(E1_FMI_B6[:,i])]
    data_counts[1,i] = data_counts[1,i] + np.count_nonzero(E1_FMI_B6[:,i][~np.isnan(E1_FMI_B6[:,i])])
    output = np.histogram(data,bins=databins, density=False)
    E1_FMI_B_6hist[:,i] = np.divide(output[:][0],np.sum(output[:][0]))
    
#Pre-feeding
E1_FMI_PF = matlab_database['FMI_exp1_PF'] #Note that the last column indicates the waiting requirement
E1_FMI_PF3 = E1_FMI_PF[E1_FMI_PF[:,-1]==3,2:-1] #Select data only for 3 s condition, dropping needless columns
E1_FMI_PF_3hist = np.zeros((databins.shape[0]-1,15)) 
for i in range(0,15): 
    data = E1_FMI_PF3[:,i][~np.isnan(E1_FMI_PF3[:,i])]
    data_counts[1,i] = data_counts[1,i] + np.count_nonzero(E1_FMI_PF3[:,i][~np.isnan(E1_FMI_PF3[:,i])])
    output = np.histogram(data,bins=databins, density=False)
    E1_FMI_PF_3hist[:,i] = np.divide(output[:][0],np.sum(output[:][0]))
    
E1_FMI_PF6 = E1_FMI_PF[E1_FMI_PF[:,-1]==6,2:-1] #Select data only for 6 s condition, dropping needless columns
E1_FMI_PF_6hist = np.zeros((databins.shape[0]-1,15)) #Empty array
for i in range(0,15):
    data = E1_FMI_PF6[:,i][~np.isnan(E1_FMI_PF6[:,i])]
    data_counts[1,i] = data_counts[1,i] + np.count_nonzero(E1_FMI_PF6[:,i][~np.isnan(E1_FMI_PF6[:,i])])
    output = np.histogram(data,bins=databins, density=False)
    E1_FMI_PF_6hist[:,i] = np.divide(output[:][0],np.sum(output[:][0]))

```

Now that the data has been sorted we can confirm the pdfs of each animal in the next cell. Note that subject 16 was dropped from analysis because the subject died before the experiment could be completed. 


```python
#Column wise group assignment mirroring the data matrices in matlab (setting this up here because it will need to be used throughout the analysis)
colwiseGA = ['LP','LP','LP','NP','NP','NP','LP','NP','NP','NP','LP','LP','LP','LP','NP','NP']
LPindices = [i for i, x in enumerate(colwiseGA) if x == "LP"]
NPindices = [i for i, x in enumerate(colwiseGA) if x == "NP"]
LPindices_mod = np.delete(LPindices,1)
NPindices = np.delete(NPindices,-1)

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig1 = plt.figure(figsize=(20,24))
k = 0
for i in range(0,15):
    plt.subplot(6,6,k+1)
    plt.plot(databins[1:len(databins)],E1_DRL_B_3hist[:,i], color = "tab:blue", linestyle = ":", linewidth = 2)
    plt.plot(databins[1:len(databins)],E1_DRL_PF_3hist[:,i], color = "tab:blue", linestyle = "-", linewidth = 2)
    plt.plot(databins[1:len(databins)],E1_DRL_B_6hist[:,i], color = "tab:orange", linestyle = ":", linewidth = 2)
    plt.plot(databins[1:len(databins)],E1_DRL_PF_6hist[:,i], color = "tab:orange", linestyle = "-", linewidth = 2)
    plt.title(colwiseGA[i]+":DRL:Subject"+str(i+1))
    plt.subplot(6,6,k+2)
    plt.plot(databins[1:len(databins)],E1_FMI_B_3hist[:,i], color = "tab:blue", linestyle = ":", linewidth = 2)
    plt.plot(databins[1:len(databins)],E1_FMI_PF_3hist[:,i], color = "tab:blue", linestyle = "-", linewidth = 2)
    plt.plot(databins[1:len(databins)],E1_FMI_B_6hist[:,i], color = "tab:orange", linestyle = ":", linewidth = 2)
    plt.plot(databins[1:len(databins)],E1_FMI_PF_6hist[:,i], color = "tab:orange", linestyle = "-", linewidth = 2)
    plt.title(colwiseGA[i]+":FMI:Subject"+str(i+1))
    k = k + 2



```


    
![png](output_7_0.png)
    


First we need to determine which version of the TR model we should use. The TR model offerred by Watterson et al. (2015) suggest there are only two states: a timing state and a non-timing state. In our expanded TR model (TRe) assumes there are three states: a timing state, a burst-response state, and a lapsed state. 

Thus, we will extract the loglikelihood of our two models across waiting requirements and feeding regimen and select the best model within each schedule. In the approach taken here we want to choose for each schedule (DRL and FMI) within each group of mice the best fitting model that captures behavior under each interval (3 and 6 s) as well as before and during the pre-feeding challenge. 


```python
#Number of free parameters
TR_params = 4 * 4 #num of params in model * number of conditions
TRe_params = 6 * 4 #num of params in model * number of conditions; note this will get multiplied by num of animals when AICc is calculcated

#Amount of data
DRLdata = data_counts[0,:]
FMIdata = data_counts[1,:]

#Preallocate matrix for loglikelihood data
DRL_e1_LLout_TR = np.empty([16,5])
DRL_e1_LLout_TRe = np.empty([16,5])
FMI_e1_LLout_TR = np.empty([16,5])
FMI_e1_LLout_TRe = np.empty([16,5])
#Preallocation matrix for AICc calculation
AICc = np.zeros([2,2])

#Grab Loglikelihood data
DRL_e1_LLout_TR[:,0] = matlab_database['LLcollector_DRL_e1_B3']
DRL_e1_LLout_TR[:,1] = matlab_database['LLcollector_DRL_e1_B6']
DRL_e1_LLout_TR[:,2] = matlab_database['LLcollector_DRL_e1_PF3']
DRL_e1_LLout_TR[:,3] = matlab_database['LLcollector_DRL_e1_PF6']
DRL_e1_LLout_TR[:,4] = np.sum(DRL_e1_LLout_TR[:,0:3],1)


DRL_e1_LLout_TRe[:,0] = matlab_database['LLcollector_DRL_e1_B3_TRe']
DRL_e1_LLout_TRe[:,1] = matlab_database['LLcollector_DRL_e1_B6_TRe']
DRL_e1_LLout_TRe[:,2] = matlab_database['LLcollector_DRL_e1_PF3_TRe']
DRL_e1_LLout_TRe[:,3] = matlab_database['LLcollector_DRL_e1_PF6_TRe']
DRL_e1_LLout_TRe[:,4] = np.sum(DRL_e1_LLout_TRe[:,0:3],1)

FMI_e1_LLout_TR[:,0] = matlab_database['LLcollector_FMI_e1_B3']
FMI_e1_LLout_TR[:,1] = matlab_database['LLcollector_FMI_e1_B6']
FMI_e1_LLout_TR[:,2] = matlab_database['LLcollector_FMI_e1_PF3']
FMI_e1_LLout_TR[:,3] = matlab_database['LLcollector_FMI_e1_PF6']
FMI_e1_LLout_TR[:,4] = np.sum(FMI_e1_LLout_TR[:,0:3],1)

FMI_e1_LLout_TRe[:,0] = matlab_database['LLcollector_FMI_e1_B3_TRe']
FMI_e1_LLout_TRe[:,1] = matlab_database['LLcollector_FMI_e1_B6_TRe']
FMI_e1_LLout_TRe[:,2] = matlab_database['LLcollector_FMI_e1_PF3_TRe']
FMI_e1_LLout_TRe[:,3] = matlab_database['LLcollector_FMI_e1_PF6_TRe']
FMI_e1_LLout_TRe[:,4] = np.sum(FMI_e1_LLout_TRe[:,0:3],1)



#Akaine Information Criterion (corrected)
AICc[0,0] = 2*(TR_params*8)+(-2*np.sum(DRL_e1_LLout_TR[LPindices,4])) + np.divide(2*((TR_params*8)^2)+(2*(TR_params*8)),np.sum(DRLdata[LPindices])-(TR_params*8)-1) #TR Model: LP animals in DRL
AICc[0,1] = 2*(TR_params*8)+(-2*np.sum(DRL_e1_LLout_TRe[LPindices,4])) + np.divide(2*((TR_params*8)^2)+(2*(TR_params*8)),np.sum(DRLdata[LPindices])-(TR_params*8)-1) #TRe Model: LP animals in DRL
AICc[1,0] = 2*(TR_params*8)+(-2*np.sum(DRL_e1_LLout_TR[NPindices,4])) + np.divide(2*((TR_params*8)^2)+(2*(TR_params*8)),np.sum(DRLdata[NPindices])-(TR_params*8)-1) #TR Model: NP animals in DRL
AICc[1,1] = 2*(TR_params*8)+(-2*np.sum(DRL_e1_LLout_TRe[NPindices,4])) + np.divide(2*((TR_params*8)^2)+(2*(TR_params*8)),np.sum(DRLdata[NPindices])-(TR_params*8)-1) #TRe Model: NP animals in DRL

AICc_DRL_dataframe = pd.DataFrame(AICc)
AICc_DRL_dataframe.columns = ["TR","TRe"]
AICc_DRL_dataframe.index = ["LP","NP"]
AICc_DRL_dataframe['minAICmodel'] = str(AICc_DRL_dataframe.idxmin(axis=1)[0])
AICc_DRL_dataframe['deltaAICc'] = AICc_DRL_dataframe.max(axis=1)-AICc_DRL_dataframe.min(axis=1) 

AICc[0,0] = 2*(TR_params*7)+(-2*np.sum(FMI_e1_LLout_TR[LPindices,4])) + np.divide(2*((TR_params*7)^2)+(2*(TR_params*7)),np.sum(DRLdata[LPindices])-(TR_params*7)-1) #TR Model: LP animals in FMI
AICc[0,1] = 2*(TR_params*7)+(-2*np.sum(FMI_e1_LLout_TRe[LPindices,4])) + np.divide(2*((TR_params*7)^2)+(2*(TR_params*7)),np.sum(DRLdata[LPindices])-(TR_params*7)-1) #TRe Model: LP animals in FMI
AICc[1,0] = 2*(TR_params*7)+(-2*np.sum(FMI_e1_LLout_TR[NPindices,4])) + np.divide(2*((TR_params*7)^2)+(2*(TR_params*7)),np.sum(DRLdata[NPindices])-(TR_params*7)-1) #TR Model: NP animals in FMI
AICc[1,1] = 2*(TR_params*7)+(-2*np.sum(FMI_e1_LLout_TRe[NPindices,4])) + np.divide(2*((TR_params*7)^2)+(2*(TR_params*7)),np.sum(DRLdata[NPindices])-(TR_params*7)-1) #TRe Model: NP animals in FMI

AICc_FMI_dataframe = pd.DataFrame(AICc)
AICc_FMI_dataframe.columns = ["TR","TRe"]
AICc_FMI_dataframe.index = ["LP","NP"]
AICc_FMI_dataframe['minAICmodel'] = str(AICc_FMI_dataframe.idxmin(axis=1)[0])
AICc_FMI_dataframe['deltaAICc'] = AICc_FMI_dataframe.max(axis=1)-AICc_FMI_dataframe.min(axis=1) 

display(AICc_DRL_dataframe)
```

    C:\Users\carte\AppData\Local\Temp/ipykernel_8420/228729674.py:55: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      AICc_DRL_dataframe['deltaAICc'] = AICc_DRL_dataframe.max(axis=1)-AICc_DRL_dataframe.min(axis=1)
    C:\Users\carte\AppData\Local\Temp/ipykernel_8420/228729674.py:66: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      AICc_FMI_dataframe['deltaAICc'] = AICc_FMI_dataframe.max(axis=1)-AICc_FMI_dataframe.min(axis=1)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TR</th>
      <th>TRe</th>
      <th>minAICmodel</th>
      <th>deltaAICc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LP</th>
      <td>30897.216060</td>
      <td>30896.764049</td>
      <td>TRe</td>
      <td>1843.897190</td>
    </tr>
    <tr>
      <th>NP</th>
      <td>25208.422674</td>
      <td>25148.596747</td>
      <td>TRe</td>
      <td>1846.228943</td>
    </tr>
  </tbody>
</table>
</div>



```python
display(AICc_FMI_dataframe)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TR</th>
      <th>TRe</th>
      <th>minAICmodel</th>
      <th>deltaAICc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LP</th>
      <td>30897.216060</td>
      <td>30896.764049</td>
      <td>TRe</td>
      <td>0.452010</td>
    </tr>
    <tr>
      <th>NP</th>
      <td>25208.422674</td>
      <td>25148.596747</td>
      <td>TRe</td>
      <td>59.825927</td>
    </tr>
  </tbody>
</table>
</div>


The minimum threshold for accepting a more complex model such as TRe is a delta AICc of 4, which corresponds to the more complex model being e^2 times more likely than the simpler model.  
The TRe model explains the data for LP and NP mice in DRL and NP in FMI.   
Interestingly, the TRe model is not necessary for capturing the data of LP mice in FMI.  
What's more, the degree of evidence for the TRe model in DRL is substantially higher (e^~900) than in FMI (when it was selected, for NP mice, e^~30).  

This suggests at least two preliminary conclusions:
1. Response withholding under DRL less temporally regulated than under FMI
2. The initiation response matters: Response withholding initiated with a nose-poke was less temporally regulated than when initiated by a LP, especially when initiation was of an FMI schedule. 

We now focus our analysis on shared parameters and derived statistics from the gamma distribution. 

Let's start by visualizing fits of the model. 


```python
#Create high rez bins for the fits
fitbins = np.linspace(0,15,31)

#First we need the parameter estimates
DRL_B3_params = matlab_database['pcollector_DRL_e1_B3_TRe']
DRL_PF3_params = matlab_database['pcollector_DRL_e1_PF3_TRe']
DRL_B6_params = matlab_database['pcollector_DRL_e1_B6_TRe']
DRL_PF6_params = matlab_database['pcollector_DRL_e1_PF6_TRe']
visDRL_B3 = np.zeros((fitbins.shape[0]-1,16)) 
visDRL_B6 = np.zeros((fitbins.shape[0]-1,16)) 
visDRL_PF3 = np.zeros((fitbins.shape[0]-1,16)) 
visDRL_PF6 = np.zeros((fitbins.shape[0]-1,16)) 
for i in range(0,16):
    cdf = DRL_B3_params[2,i]*gamma.cdf(fitbins[1:len(fitbins)],1+DRL_B3_params[0,i],0,DRL_B3_params[1,i])+DRL_B3_params[4,i]*expon.cdf(fitbins[1:len(fitbins)],0,1/DRL_B3_params[3,i])+(1-DRL_B3_params[4,i]-DRL_B3_params[2,i])*expon.cdf(fitbins[1:len(fitbins)],0,1/DRL_B3_params[5,i])
    hist = cdf[0]
    hist = np.append(hist,np.diff(cdf))
    visDRL_B3[:,i] = hist
    
    cdf = DRL_B6_params[2,i]*gamma.cdf(fitbins[1:len(fitbins)],1+DRL_B6_params[0,i],0,DRL_B6_params[1,i])+DRL_B6_params[4,i]*expon.cdf(fitbins[1:len(fitbins)],0,1/DRL_B6_params[3,i])+(1-DRL_B6_params[4,i]-DRL_B6_params[2,i])*expon.cdf(fitbins[1:len(fitbins)],0,1/DRL_B6_params[5,i])
    hist = cdf[0]
    hist = np.append(hist,np.diff(cdf))
    visDRL_B6[:,i] = hist
    
    cdf = DRL_PF3_params[2,i]*gamma.cdf(fitbins[1:len(fitbins)],1+DRL_PF3_params[0,i],0,DRL_PF3_params[1,i])+DRL_PF3_params[4,i]*expon.cdf(fitbins[1:len(fitbins)],0,1/DRL_PF3_params[3,i])+(1-DRL_PF3_params[4,i]-DRL_PF3_params[2,i])*expon.cdf(fitbins[1:len(fitbins)],0,1/DRL_PF3_params[5,i])
    hist = cdf[0]
    hist = np.append(hist,np.diff(cdf))
    visDRL_PF3[:,i] = hist
    
    cdf = DRL_PF6_params[2,i]*gamma.cdf(fitbins[1:len(fitbins)],1+DRL_PF6_params[0,i],0,DRL_PF6_params[1,i])+DRL_PF6_params[4,i]*expon.cdf(fitbins[1:len(fitbins)],0,1/DRL_PF6_params[3,i])+(1-DRL_PF6_params[4,i]-DRL_PF6_params[2,i])*expon.cdf(fitbins[1:len(fitbins)],0,1/DRL_PF6_params[5,i])
    hist = cdf[0]
    hist = np.append(hist,np.diff(cdf))
    visDRL_PF6[:,i] = hist
    
#First we need the parameter estimates

#First grab TRe only
FMI_B3_params = matlab_database['pcollector_FMI_e1_B3_TRe']
FMI_PF3_params = matlab_database['pcollector_FMI_e1_PF3_TRe']
FMI_B6_params = matlab_database['pcollector_FMI_e1_B6_TRe']
FMI_PF6_params = matlab_database['pcollector_FMI_e1_PF6_TRe']

#Then replace LP with TR only
#First replace rows 5:6 wit
FMI_B3_paramsTR = matlab_database['pcollector_FMI_e1_B3']
FMI_PF3_paramsTR = matlab_database['pcollector_FMI_e1_PF3']
FMI_B6_paramsTR = matlab_database['pcollector_FMI_e1_B6']
FMI_PF6_paramsTR = matlab_database['pcollector_FMI_e1_PF6']

FMI_B3_params[0:3,LPindices] = FMI_B3_paramsTR[0:3,LPindices]
FMI_PF3_params[0:3,LPindices] = FMI_PF3_paramsTR[0:3,LPindices]
FMI_B6_params[0:3,LPindices] = FMI_B6_paramsTR[0:3,LPindices]
FMI_PF6_params[0:3,LPindices] = FMI_PF6_paramsTR[0:3,LPindices]

FMI_B3_params[4:FMI_B3_params.shape[1],LPindices] = np.nan
FMI_PF3_params[4:FMI_B3_params.shape[1],LPindices] = np.nan
FMI_B6_params[4:FMI_B3_params.shape[1],LPindices] = np.nan
FMI_PF6_params[4:FMI_B3_params.shape[1],LPindices] = np.nan

visFMI_B3 = np.zeros((fitbins.shape[0]-1,15)) 
visFMI_B6 = np.zeros((fitbins.shape[0]-1,15)) 
visFMI_PF3 = np.zeros((fitbins.shape[0]-1,15)) 
visFMI_PF6 = np.zeros((fitbins.shape[0]-1,15)) 
for i in range(0,15):
    if i in LPindices:
        cdf = FMI_B3_params[2,i]*gamma.cdf(fitbins[1:len(fitbins)],1+FMI_B3_params[0,i],0,FMI_B3_params[1,i])+(1-FMI_B3_params[2,i])*expon.cdf(fitbins[1:len(fitbins)],0,1/FMI_B3_params[3,i])
        hist = cdf[0]
        hist = np.append(hist,np.diff(cdf))
        visFMI_B3[:,i] = hist
    
        cdf = FMI_B6_params[2,i]*gamma.cdf(fitbins[1:len(fitbins)],1+FMI_B6_params[0,i],0,FMI_B6_params[1,i])+(1-FMI_B6_params[2,i])*expon.cdf(fitbins[1:len(fitbins)],0,1/FMI_B6_params[3,i])
        hist = cdf[0]
        hist = np.append(hist,np.diff(cdf))
        visFMI_B6[:,i] = hist
    
        cdf = FMI_PF3_params[2,i]*gamma.cdf(fitbins[1:len(fitbins)],1+FMI_PF3_params[0,i],0,FMI_PF3_params[1,i])+(1-FMI_PF3_params[2,i])*expon.cdf(fitbins[1:len(fitbins)],0,1/FMI_PF3_params[3,i])
        hist = cdf[0]
        hist = np.append(hist,np.diff(cdf))
        visFMI_PF3[:,i] = hist
    
        cdf = FMI_PF6_params[2,i]*gamma.cdf(fitbins[1:len(fitbins)],1+FMI_PF6_params[0,i],0,FMI_PF6_params[1,i])+(1-FMI_PF6_params[2,i])*expon.cdf(fitbins[1:len(fitbins)],0,1/FMI_PF6_params[3,i])
        hist = cdf[0]
        hist = np.append(hist,np.diff(cdf))
        visFMI_PF6[:,i] = hist
    else:
        cdf = FMI_B3_params[2,i]*gamma.cdf(fitbins[1:len(fitbins)],1+FMI_B3_params[0,i],0,FMI_B3_params[1,i])+FMI_B3_params[4,i]*expon.cdf(fitbins[1:len(fitbins)],0,1/FMI_B3_params[3,i])+(1-FMI_B3_params[4,i]-FMI_B3_params[2,i])*expon.cdf(fitbins[1:len(fitbins)],0,1/FMI_B3_params[5,i])
        hist = cdf[0]
        hist = np.append(hist,np.diff(cdf))
        visFMI_B3[:,i] = hist
    
        cdf = FMI_B6_params[2,i]*gamma.cdf(fitbins[1:len(fitbins)],1+FMI_B6_params[0,i],0,FMI_B6_params[1,i])+FMI_B6_params[4,i]*expon.cdf(fitbins[1:len(fitbins)],0,1/FMI_B6_params[3,i])+(1-FMI_B6_params[4,i]-FMI_B6_params[2,i])*expon.cdf(fitbins[1:len(fitbins)],0,1/FMI_B6_params[5,i])
        hist = cdf[0]
        hist = np.append(hist,np.diff(cdf))
        visFMI_B6[:,i] = hist
    
        cdf = FMI_PF3_params[2,i]*gamma.cdf(fitbins[1:len(fitbins)],1+FMI_PF3_params[0,i],0,FMI_PF3_params[1,i])+FMI_PF3_params[4,i]*expon.cdf(fitbins[1:len(fitbins)],0,1/FMI_PF3_params[3,i])+(1-FMI_PF3_params[4,i]-FMI_PF3_params[2,i])*expon.cdf(fitbins[1:len(fitbins)],0,1/FMI_PF3_params[5,i])
        hist = cdf[0]
        hist = np.append(hist,np.diff(cdf))
        visFMI_PF3[:,i] = hist
    
        cdf = FMI_PF6_params[2,i]*gamma.cdf(fitbins[1:len(fitbins)],1+FMI_PF6_params[0,i],0,FMI_PF6_params[1,i])+FMI_PF6_params[4,i]*expon.cdf(fitbins[1:len(fitbins)],0,1/FMI_PF6_params[3,i])+(1-FMI_PF6_params[4,i]-FMI_PF6_params[2,i])*expon.cdf(fitbins[1:len(fitbins)],0,1/FMI_PF6_params[5,i])
        hist = cdf[0]
        hist = np.append(hist,np.diff(cdf))
        visFMI_PF6[:,i] = hist        
    
```


```python
DRL_B3_params
```




    array([[2.72273971e+01, 5.41876869e-01, 3.01089827e+01, 5.06048132e+00,
            6.76876553e+00, 1.74687296e+01, 1.20196669e+01, 6.06999056e+00,
            2.51752291e+01, 1.57597143e+01, 1.87203243e+01, 1.55641602e+00,
            1.55375069e+01, 1.60116117e+01, 1.32238674e+01, 2.93481343e+01],
           [1.09529162e-01, 1.15323945e+00, 1.00744040e-01, 4.91272817e-01,
            4.78174906e-01, 1.51794183e-01, 2.59897472e-01, 4.82746565e-01,
            1.34191594e-01, 1.97178228e-01, 1.56287534e-01, 6.91937406e-01,
            1.72417113e-01, 1.80125673e-01, 2.10300923e-01, 1.02302789e-01],
           [7.52747595e-01, 7.45991416e-01, 6.75556966e-01, 4.22273252e-01,
            3.81895879e-01, 2.44215835e-01, 9.34975343e-01, 2.96115502e-01,
            4.71839824e-01, 3.84486342e-01, 3.95434347e-01, 9.12203184e-01,
            5.12301068e-01, 5.57065675e-01, 4.37275750e-01, 3.97227658e-01],
           [2.24793411e+00, 3.23330087e+00, 2.10548998e+00, 8.89053268e-01,
            1.48969879e+00, 1.31489326e+00, 5.77142588e+00, 1.38554435e+00,
            3.75103565e-01, 2.52941646e-01, 1.99207857e+00, 2.63197308e+00,
            3.25208831e+00, 2.41221906e+00, 2.29493805e-01, 4.65084238e-01],
           [2.44309779e-01, 2.49461401e-01, 3.00312743e-01, 5.28158174e-01,
            5.09012982e-01, 6.16540911e-01, 6.26851594e-02, 6.95839488e-01,
            4.13377301e-01, 5.93336580e-01, 5.89541658e-01, 8.60041416e-02,
            4.84930312e-01, 2.82091957e-01, 5.00007848e-01, 5.19746810e-01],
           [1.01728772e+01, 1.78321117e+01, 1.61338221e+01, 6.20204650e+00,
            3.71369282e+00, 1.31489452e+00, 1.60753616e+01, 2.27770968e+01,
            3.97660892e+00, 2.79100355e+01, 3.02460385e+01, 2.75296450e+02,
            1.42731091e+01, 6.00966440e+00, 5.58711495e+00, 5.38020125e+00]])



Now we can visualize the fit of the TR model to the constructed probability functions for each animal


```python

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig1 = plt.figure(figsize=(20,24))
k = 0
for i in range(0,15):
    plt.subplot(6,6,k+1)
    plt.plot(databins[1:len(databins)],E1_DRL_B_3hist[:,i], color = "tab:blue",  marker = "s", linestyle="None")
    plt.plot(databins[1:len(databins)],E1_DRL_PF_3hist[:,i], color = "tab:blue", marker = "o", linestyle="None")
    plt.plot(databins[1:len(databins)],E1_DRL_B_6hist[:,i], color = "tab:orange", marker = "s", linestyle="None")
    plt.plot(databins[1:len(databins)],E1_DRL_PF_6hist[:,i], color = "tab:orange", marker = "o", linestyle="None")
    plt.plot(fitbins[1:len(fitbins)], visDRL_B3[:,i], color = "tab:blue", linestyle="-")
    plt.plot(fitbins[1:len(fitbins)], visDRL_PF3[:,i], color = "tab:blue", linestyle=":")
    plt.plot(fitbins[1:len(fitbins)], visDRL_B6[:,i], color = "tab:orange", linestyle="-")
    plt.plot(fitbins[1:len(fitbins)], visDRL_PF6[:,i], color = "tab:orange", linestyle=":")
    plt.title(colwiseGA[i]+":DRL:Subject"+str(i+1))
    plt.subplot(6,6,k+2)
    plt.plot(databins[1:len(databins)],E1_FMI_B_3hist[:,i], color = "tab:blue", marker="s", linestyle="None")
    plt.plot(databins[1:len(databins)],E1_FMI_PF_3hist[:,i], color = "tab:blue", marker = "o", linestyle="None")
    plt.plot(databins[1:len(databins)],E1_FMI_B_6hist[:,i], color = "tab:orange",  marker="s", linestyle="None")
    plt.plot(databins[1:len(databins)],E1_FMI_PF_6hist[:,i], color = "tab:orange",marker = "o", linestyle="None")
    plt.plot(fitbins[1:len(fitbins)], visFMI_B3[:,i], color = "tab:blue", linestyle="-")
    plt.plot(fitbins[1:len(fitbins)], visFMI_PF3[:,i], color = "tab:blue", linestyle=":")
    plt.plot(fitbins[1:len(fitbins)], visFMI_B6[:,i], color = "tab:orange", linestyle="-")
    plt.plot(fitbins[1:len(fitbins)], visFMI_PF6[:,i], color = "tab:orange", linestyle=":")
    plt.title(colwiseGA[i]+":FMI:Subject"+str(i+1))
    k = k + 2
```


    
![png](output_15_0.png)
    


Now that we've calculated the predicted PDFs we can look at how well it describes the average PDF for each group under each condition.


```python
#Sort data
LP_mean_DRL_PDFs = [np.nanmean(E1_DRL_B_3hist[:,LPindices],axis=1),np.nanmean(E1_DRL_PF_3hist[:,LPindices],axis=1),np.nanmean(E1_DRL_B_6hist[:,LPindices],axis=1),np.nanmean(E1_DRL_PF_6hist[:,LPindices_mod],axis=1)]
LP_mean_FMI_PDFs = [np.nanmean(E1_FMI_B_3hist[:,LPindices],axis=1),np.nanmean(E1_FMI_PF_3hist[:,LPindices],axis=1),np.nanmean(E1_FMI_B_6hist[:,LPindices],axis=1),np.nanmean(E1_FMI_PF_6hist[:,LPindices],axis=1)]
LP_mean_DRL_predPDFs = [np.nanmean(visDRL_B3[:,LPindices],axis=1),np.nanmean(visDRL_PF3[:,LPindices],axis=1),np.nanmean(visDRL_B6[:,LPindices],axis=1),np.nanmean(visDRL_PF6[:,LPindices_mod],axis=1)]
LP_mean_FMI_predPDFs = [np.nanmean(visFMI_B3[:,LPindices],axis=1),np.nanmean(visFMI_PF3[:,LPindices],axis=1),np.nanmean(visFMI_B6[:,LPindices],axis=1),np.nanmean(visFMI_PF6[:,LPindices],axis=1)]

NP_mean_DRL_PDFs = [np.nanmean(E1_DRL_B_3hist[:,NPindices],axis=1),np.nanmean(E1_DRL_PF_3hist[:,NPindices],axis=1),np.nanmean(E1_DRL_B_6hist[:,NPindices],axis=1),np.nanmean(E1_DRL_PF_6hist[:,NPindices],axis=1)]
NP_mean_FMI_PDFs = [np.nanmean(E1_FMI_B_3hist[:,NPindices],axis=1),np.nanmean(E1_FMI_PF_3hist[:,NPindices],axis=1),np.nanmean(E1_FMI_B_6hist[:,NPindices],axis=1),np.nanmean(E1_FMI_PF_6hist[:,NPindices],axis=1)]
NP_mean_DRL_predPDFs = [np.nanmean(visDRL_B3[:,NPindices],axis=1),np.nanmean(visDRL_PF3[:,NPindices],axis=1),np.nanmean(visDRL_B6[:,NPindices],axis=1),np.nanmean(visDRL_PF6[:,NPindices],axis=1)]
NP_mean_FMI_predPDFs = [np.nanmean(visFMI_B3[:,NPindices],axis=1),np.nanmean(visFMI_PF3[:,NPindices],axis=1),np.nanmean(visFMI_B6[:,NPindices],axis=1),np.nanmean(visFMI_PF6[:,NPindices],axis=1)]


#Plot data
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig2 = plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
plt.plot(databins[1:len(databins)],LP_mean_DRL_PDFs[0], color = "tab:blue", linestyle="None", marker = "s",label = '3-s Baseline')
plt.plot(databins[1:len(databins)],LP_mean_DRL_PDFs[1], color = "tab:blue", linestyle="None", marker = "o",label = '3-s Pre-feeding',markerfacecolor="white",markeredgecolor='tab:blue', markeredgewidth=1)
plt.plot(databins[1:len(databins)],LP_mean_DRL_PDFs[2], color = "tab:orange", linestyle="None", marker = "s",label = '6-s Baseline')
plt.plot(databins[1:len(databins)],LP_mean_DRL_PDFs[3], color = "tab:orange", linestyle="None", marker = "o",label = '6-s Pre-feeding',markerfacecolor="white",markeredgecolor='tab:orange', markeredgewidth=1)
plt.plot(fitbins[1:len(fitbins)],LP_mean_DRL_predPDFs[0], color = "tab:blue", linestyle="-", linewidth=2,label = '3-s Baseline Fit')
plt.plot(fitbins[1:len(fitbins)],LP_mean_DRL_predPDFs[1], color = "tab:blue", linestyle=":", linewidth=2,label = '3-s Pre-feeding Fit')
plt.plot(fitbins[1:len(fitbins)],LP_mean_DRL_predPDFs[2], color = "tab:orange", linestyle="-", linewidth=2,label = '6-s Baseline Fit')
plt.plot(fitbins[1:len(fitbins)],LP_mean_DRL_predPDFs[3], color = "tab:orange", linestyle=":", linewidth=2,label = '6-s Pre-feeding Fit')
plt.title('LP DRL')
plt.legend()

plt.ylabel('P(Inter-Response Time = X)', fontsize = 14)
plt.subplot(2,2,3)
plt.plot(databins[1:len(databins)],LP_mean_FMI_PDFs[0], color = "tab:blue", linestyle="None", marker = "s")
plt.plot(databins[1:len(databins)],LP_mean_FMI_PDFs[1], color = "tab:blue", linestyle="None", marker = "o",markerfacecolor="white",markeredgecolor='tab:blue', markeredgewidth=1)
plt.plot(databins[1:len(databins)],LP_mean_FMI_PDFs[2], color = "tab:orange", linestyle="None", marker = "s")
plt.plot(databins[1:len(databins)],LP_mean_FMI_PDFs[3], color = "tab:orange", linestyle="None", marker = "o",markerfacecolor="white",markeredgecolor='tab:orange', markeredgewidth=1)
plt.plot(fitbins[1:len(fitbins)],LP_mean_FMI_predPDFs[0], color = "tab:blue", linestyle="-", linewidth=2)
plt.plot(fitbins[1:len(fitbins)],LP_mean_FMI_predPDFs[1], color = "tab:blue", linestyle=":", linewidth=2)
plt.plot(fitbins[1:len(fitbins)],LP_mean_FMI_predPDFs[2], color = "tab:orange", linestyle="-", linewidth=2)
plt.plot(fitbins[1:len(fitbins)],LP_mean_FMI_predPDFs[3], color = "tab:orange", linestyle=":", linewidth=2)
plt.title('LP FMI')


plt.subplot(2,2,2)
plt.plot(databins[1:len(databins)],NP_mean_DRL_PDFs[0], color = "tab:blue", linestyle="None", marker = "s")
plt.plot(databins[1:len(databins)],NP_mean_DRL_PDFs[1], color = "tab:blue", linestyle="None", marker = "o",markerfacecolor="white",markeredgecolor='tab:blue', markeredgewidth=1)
plt.plot(databins[1:len(databins)],NP_mean_DRL_PDFs[2], color = "tab:orange", linestyle="None", marker = "s")
plt.plot(databins[1:len(databins)],NP_mean_DRL_PDFs[3], color = "tab:orange", linestyle="None", marker = "o",markerfacecolor="white",markeredgecolor='tab:orange', markeredgewidth=1)
plt.plot(fitbins[1:len(fitbins)],NP_mean_DRL_predPDFs[0], color = "tab:blue", linestyle="-", linewidth=2)
plt.plot(fitbins[1:len(fitbins)],NP_mean_DRL_predPDFs[1], color = "tab:blue", linestyle=":", linewidth=2)
plt.plot(fitbins[1:len(fitbins)],NP_mean_DRL_predPDFs[2], color = "tab:orange", linestyle="-", linewidth=2)
plt.plot(fitbins[1:len(fitbins)],NP_mean_DRL_predPDFs[3], color = "tab:orange", linestyle=":", linewidth=2)
plt.ylabel('P(Inter-Response Time = X)', fontsize = 14)
plt.xlabel('Inter-Response Time (s)', fontsize = 14)
plt.title('NP DRL')


plt.subplot(2,2,4)
plt.plot(databins[1:len(databins)],NP_mean_FMI_PDFs[0], color = "tab:blue", linestyle="None", marker = "s")
plt.plot(databins[1:len(databins)],NP_mean_FMI_PDFs[1], color = "tab:blue", linestyle="None", marker = "o",markerfacecolor="white",markeredgecolor='tab:blue', markeredgewidth=1)
plt.plot(databins[1:len(databins)],NP_mean_FMI_PDFs[2], color = "tab:orange", linestyle="None", marker = "s")
plt.plot(databins[1:len(databins)],NP_mean_FMI_PDFs[3], color = "tab:orange", linestyle="None", marker = "o",markerfacecolor="white",markeredgecolor='tab:orange', markeredgewidth=1)
plt.plot(fitbins[1:len(fitbins)],NP_mean_FMI_predPDFs[0], color = "tab:blue", linestyle="-", linewidth=2)
plt.plot(fitbins[1:len(fitbins)],NP_mean_FMI_predPDFs[1], color = "tab:blue", linestyle=":", linewidth=2)
plt.plot(fitbins[1:len(fitbins)],NP_mean_FMI_predPDFs[2], color = "tab:orange", linestyle="-", linewidth=2)
plt.plot(fitbins[1:len(fitbins)],NP_mean_FMI_predPDFs[3], color = "tab:orange", linestyle=":", linewidth=2)
plt.xlabel('Inter-Response Time (s)', fontsize = 14)
plt.title('NP FMI')


```




    Text(0.5, 1.0, 'NP FMI')




    
![png](output_17_1.png)
    


Now that we've visualized the fit of our model let's analyze model parameters and derived statistics. 

First we will analyze derived statistics of the gamma distribution:
* mu = the mean timed response
* sd = variance of timed responses
* cv = sd/mu

Second we will analyze parameters common to all groups of mice in all conditions. 
* P = the probability of timing
* q (in TRe) and 1-P (in TR) = the probability of burst responding
* K_burst = the mean burst response time

Then we will analyze parameters common to only mice with the TRe model
* 1-q-P = the probability of lapsed responding
* K_lapse = the mean lapse response time



```python
#parameters
all_params=pd.DataFrame(np.concatenate((DRL_B3_params,DRL_PF3_params,DRL_B6_params,DRL_PF6_params,FMI_B3_params,FMI_PF3_params,FMI_B6_params,FMI_PF6_params),axis=0))
all_params=all_params.set_axis({"LPM1","LPM2","LPM3","NPM4","NPM5","NPM6","LPM7","NPM8","NPM9","NPM10","LPM11","LPM12","LPM13","LPM14","NPM15","NPM16"},axis=1)
all_params['Params'] = np.tile(['epsilon','c','P','Kburst','q','Klapse'],8)
all_params['Interval'] = np.repeat([3,3,6,6,3,3,6,6],6)
all_params['Motivation'] = np.repeat(['B','PF','B','PF','B','PF','B','PF'],6)
all_params['Schedule'] = np.repeat(['DRL','DRL','DRL','DRL','FMI','FMI','FMI','FMI'],6)
all_params = pd.melt(all_params,id_vars = ['Params','Interval','Motivation','Schedule'])
all_params['MouseID'] = 0
all_params['InitRsp'] = "NAN"
id_var, init_var = [],[]
for lbl in all_params.variable:
    id_var.append(int(re.findall(r'\d+',lbl)[0]))
    init_var.append(re.findall("[a-zA-Z]+",lbl)[0])
all_params['MouseID'] = id_var
all_params['InitRsp'] = init_var
all_params
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Params</th>
      <th>Interval</th>
      <th>Motivation</th>
      <th>Schedule</th>
      <th>variable</th>
      <th>value</th>
      <th>MouseID</th>
      <th>InitRsp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>epsilon</td>
      <td>3</td>
      <td>B</td>
      <td>DRL</td>
      <td>LPM1</td>
      <td>27.227397</td>
      <td>1</td>
      <td>LPM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c</td>
      <td>3</td>
      <td>B</td>
      <td>DRL</td>
      <td>LPM1</td>
      <td>0.109529</td>
      <td>1</td>
      <td>LPM</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P</td>
      <td>3</td>
      <td>B</td>
      <td>DRL</td>
      <td>LPM1</td>
      <td>0.752748</td>
      <td>1</td>
      <td>LPM</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kburst</td>
      <td>3</td>
      <td>B</td>
      <td>DRL</td>
      <td>LPM1</td>
      <td>2.247934</td>
      <td>1</td>
      <td>LPM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>q</td>
      <td>3</td>
      <td>B</td>
      <td>DRL</td>
      <td>LPM1</td>
      <td>0.244310</td>
      <td>1</td>
      <td>LPM</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>c</td>
      <td>6</td>
      <td>PF</td>
      <td>FMI</td>
      <td>NPM10</td>
      <td>1.211060</td>
      <td>10</td>
      <td>NPM</td>
    </tr>
    <tr>
      <th>764</th>
      <td>P</td>
      <td>6</td>
      <td>PF</td>
      <td>FMI</td>
      <td>NPM10</td>
      <td>0.266228</td>
      <td>10</td>
      <td>NPM</td>
    </tr>
    <tr>
      <th>765</th>
      <td>Kburst</td>
      <td>6</td>
      <td>PF</td>
      <td>FMI</td>
      <td>NPM10</td>
      <td>1.313623</td>
      <td>10</td>
      <td>NPM</td>
    </tr>
    <tr>
      <th>766</th>
      <td>q</td>
      <td>6</td>
      <td>PF</td>
      <td>FMI</td>
      <td>NPM10</td>
      <td>0.151613</td>
      <td>10</td>
      <td>NPM</td>
    </tr>
    <tr>
      <th>767</th>
      <td>Klapse</td>
      <td>6</td>
      <td>PF</td>
      <td>FMI</td>
      <td>NPM10</td>
      <td>8.116420</td>
      <td>10</td>
      <td>NPM</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 8 columns</p>
</div>




```python
#derived statistics
all_derivedstats = pd.DataFrame()
all_derivedstats['mu'] = (all_params.value[all_params['Params']=='epsilon'].to_numpy()+1)*all_params.value[all_params['Params']=='c'].to_numpy()
all_derivedstats['sd'] = np.sqrt((all_params.value[all_params['Params']=='epsilon'].to_numpy()+1)*np.power(all_params.value[all_params['Params']=='c'].to_numpy(),2))
all_derivedstats['cv'] = all_derivedstats['sd'].to_numpy()/all_derivedstats['mu'].to_numpy()
all_derivedstats['Schedule'] = all_params.Schedule[all_params.Params=='epsilon'].tolist()
all_derivedstats['Motivation'] = all_params.Motivation[all_params.Params=='epsilon'].tolist()
all_derivedstats['Interval'] = all_params.Interval[all_params.Params=='epsilon'].tolist()
all_derivedstats['MouseID'] = all_params.MouseID[all_params.Params=='epsilon'].tolist()
all_derivedstats['InitRsp'] = all_params.InitRsp[all_params.Params=='epsilon'].tolist()
all_derivedstats = all_derivedstats.melt(id_vars=['Schedule','Motivation','Interval','MouseID','InitRsp'])
all_derivedstats = all_derivedstats[all_derivedstats['MouseID']<=15]
all_derivedstats['lnValue'] = np.log(all_derivedstats.value)
all_derivedstats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Schedule</th>
      <th>Motivation</th>
      <th>Interval</th>
      <th>MouseID</th>
      <th>InitRsp</th>
      <th>variable</th>
      <th>value</th>
      <th>lnValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DRL</td>
      <td>B</td>
      <td>3</td>
      <td>1</td>
      <td>LPM</td>
      <td>mu</td>
      <td>3.091723</td>
      <td>1.128729</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DRL</td>
      <td>PF</td>
      <td>3</td>
      <td>1</td>
      <td>LPM</td>
      <td>mu</td>
      <td>3.303817</td>
      <td>1.195078</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DRL</td>
      <td>B</td>
      <td>6</td>
      <td>1</td>
      <td>LPM</td>
      <td>mu</td>
      <td>5.712226</td>
      <td>1.742609</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DRL</td>
      <td>PF</td>
      <td>6</td>
      <td>1</td>
      <td>LPM</td>
      <td>mu</td>
      <td>6.243803</td>
      <td>1.831590</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FMI</td>
      <td>B</td>
      <td>3</td>
      <td>1</td>
      <td>LPM</td>
      <td>mu</td>
      <td>3.150011</td>
      <td>1.147406</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>379</th>
      <td>DRL</td>
      <td>PF</td>
      <td>6</td>
      <td>10</td>
      <td>NPM</td>
      <td>cv</td>
      <td>0.174283</td>
      <td>-1.747077</td>
    </tr>
    <tr>
      <th>380</th>
      <td>FMI</td>
      <td>B</td>
      <td>3</td>
      <td>10</td>
      <td>NPM</td>
      <td>cv</td>
      <td>0.469293</td>
      <td>-0.756529</td>
    </tr>
    <tr>
      <th>381</th>
      <td>FMI</td>
      <td>PF</td>
      <td>3</td>
      <td>10</td>
      <td>NPM</td>
      <td>cv</td>
      <td>0.469293</td>
      <td>-0.756529</td>
    </tr>
    <tr>
      <th>382</th>
      <td>FMI</td>
      <td>B</td>
      <td>6</td>
      <td>10</td>
      <td>NPM</td>
      <td>cv</td>
      <td>0.469293</td>
      <td>-0.756529</td>
    </tr>
    <tr>
      <th>383</th>
      <td>FMI</td>
      <td>PF</td>
      <td>6</td>
      <td>10</td>
      <td>NPM</td>
      <td>cv</td>
      <td>0.469293</td>
      <td>-0.756529</td>
    </tr>
  </tbody>
</table>
<p>360 rows × 8 columns</p>
</div>




```python
#Analysis of derived statistics; graphing then analysis
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig3 = plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
sns.barplot(x="Interval",y="lnValue",hue="Motivation",ci=68,data=all_derivedstats[((all_derivedstats['Schedule']=="DRL") & (all_derivedstats['variable']=="mu") & (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('log(Gamma Mean (s))', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Nose Poke", fontsize=14)
plt.subplot(2,2,2)
sns.barplot(x="Interval",y="lnValue",hue="Motivation",ci=68,data=all_derivedstats[((all_derivedstats['Schedule']=="DRL") & (all_derivedstats['variable']=="mu")& (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('log(Gamma Mean (s))', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Lever Press", fontsize=14)
plt.subplot(2,2,3)
sns.barplot(x="Interval",y="lnValue",hue="Motivation",ci=68,data=all_derivedstats[((all_derivedstats['Schedule']=="FMI") & (all_derivedstats['variable']=="mu")& (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('log(Gamma Mean (s))', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Nose Poke", fontsize=14)
plt.subplot(2,2,4)
sns.barplot(x="Interval",y="lnValue",hue="Motivation",ci=68,data=all_derivedstats[((all_derivedstats['Schedule']=="FMI") & (all_derivedstats['variable']=="mu")& (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('log(Gamma Mean (s))', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Lever Press", fontsize=14)
plt.tight_layout()
```


    
![png](output_21_0.png)
    



```python
#Analysis of derived statistics; graphing then analysis
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig3 = plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
sns.barplot(x="Interval",y="lnValue",hue="Motivation",ci=95,data=all_derivedstats[((all_derivedstats['Schedule']=="DRL") & (all_derivedstats['variable']=="sd") & (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('log(Gamma SD (s))', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Nose Poke", fontsize=14)
plt.subplot(2,2,2)
sns.barplot(x="Interval",y="lnValue",hue="Motivation",ci=95,data=all_derivedstats[((all_derivedstats['Schedule']=="DRL") & (all_derivedstats['variable']=="sd") & (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('log(Gamma SD (s))', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Lever Press", fontsize=14)
plt.subplot(2,2,3)
sns.barplot(x="Interval",y="lnValue",hue="Motivation",ci=95,data=all_derivedstats[((all_derivedstats['Schedule']=="FMI") & (all_derivedstats['variable']=="sd") & (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('log(Gamma SD (s))', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Nose Poke", fontsize=14)
plt.subplot(2,2,4)
sns.barplot(x="Interval",y="lnValue",hue="Motivation",ci=95,data=all_derivedstats[((all_derivedstats['Schedule']=="FMI") & (all_derivedstats['variable']=="sd") & (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('log(Gamma SD (s))', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Lever Press", fontsize=14)
plt.tight_layout()
```


    
![png](output_22_0.png)
    



```python
#Analysis of derived statistics; graphing then analysis
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig3 = plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_derivedstats[((all_derivedstats['Schedule']=="DRL") & (all_derivedstats['variable']=="cv") & (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('Gamma CV', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Nose Poke", fontsize=14)
plt.subplot(2,2,2)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_derivedstats[((all_derivedstats['Schedule']=="DRL") & (all_derivedstats['variable']=="cv") & (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('Gamma CV', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Lever Press", fontsize=14)
plt.subplot(2,2,3)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_derivedstats[((all_derivedstats['Schedule']=="FMI") & (all_derivedstats['variable']=="cv") & (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('Gamma CV', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Nose Poke", fontsize=14)
plt.subplot(2,2,4)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_derivedstats[((all_derivedstats['Schedule']=="FMI") & (all_derivedstats['variable']=="cv") & (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('Gamma CV', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Lever Press", fontsize=14)
plt.tight_layout()
```


    
![png](output_23_0.png)
    


To analyze the above graphs data will be analyzed in the following R file: see topography_gammaanalysis.Rmd



```python
all_derivedstats.to_csv("topography_gamma.csv")

```

The output of the analysis conducted in R can be found as topograph_gamma.Rmd and .html in the Github repo. Here I summarize the main findings. Note that because mice were trained in sequence from DRL to FMI  

1. The mean of gamma distributed IRTs was sensitive to a main effect of interval and interaction of schedule and motivation (log-BF10 = 6.15). This indicates  that the mean of gamma distributed IRTs scaled with the waiting requirement (3-->6). Additionally, it indicates the mean of gamma distributed IRTs was sensitive differentially sensitive to motivation based on whether mice were performing in DRL or FMI. Probes of this interaction revealed that whereas pre-feeding increased the mean gamma distributed IRT in DRL (log-BF10 = 1.26), it had no effect in FMI (log-BF10 = -0.84). 
2. The sd of gamma distributed IRTs was only sensitive to the interval (log-BF10 = 5.18). This indicates that the sd of gamma distributed IRTs scaled with the waiting requirement. Taken together with the scaling of the mean of gamma distributed IRTs it suggests gamma distributed IRTs are scalar invariant. 
3. The CV of gamma distributed IRTs was not sensitive to manipulation, thus confirming the scalar property (largest log-BF10 = -0.51). 
4. Relative to the effects reported above and to the null model, there was no strong evidence for initiating response affecting gamma distributed IRTs (largest log-BF10 = -0.58). 


Taken together, these data suggest that the mean of gamma distributed IRTs is sensitive to flucutations in motivation when mice are trained in DRL but not FMI. We also replicated the typical finding that within a restricted range of intervals (seconds-minutes) timing performance is scalar invariant. Gamma distributed IRTs were not sensitive to initiating response. 

Next, we analyzed the remaining parameters of the mixture models. Note that whereas analyzing parameters of the gamma can help us map effects onto aspects of a pacemaker-accumulator model of interval timing (e.g., epsilon: threshold; c: clock-speed), the remaining parameters will help us characterize how frequently mice burst or lapse respond and the length of the IRTs generated under those modes. 


```python
#add ln of values because not present 
all_params['lnValue']=1 #just establishing column because need to log-odds probabilities
all_params.lnValue[all_params['Params']=='epsilon']=np.log(all_params.value[all_params['Params']=="epsilon"])
all_params.lnValue[all_params['Params']=='c']=np.log(all_params.value[all_params['Params']=="c"])
all_params.lnValue[all_params['Params']=='Kburst']=np.log(all_params.value[all_params['Params']=="Kburst"])
all_params.lnValue[all_params['Params']=='Klapse']=np.log(all_params.value[all_params['Params']=="Klapse"])
all_params.lnValue[all_params['Params']=='P']=np.log((all_params.value[all_params['Params']=="P"]/(1-all_params.value[all_params['Params']=="P"])))
all_params.lnValue[all_params['Params']=='q']=np.log((all_params.value[all_params['Params']=="q"]/(1-all_params.value[all_params['Params']=="q"])))

```

    C:\Users\carte\AppData\Local\Temp/ipykernel_8420/1061268490.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      all_params.lnValue[all_params['Params']=='epsilon']=np.log(all_params.value[all_params['Params']=="epsilon"])
    C:\Users\carte\AppData\Local\Temp/ipykernel_8420/1061268490.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      all_params.lnValue[all_params['Params']=='c']=np.log(all_params.value[all_params['Params']=="c"])
    C:\Users\carte\AppData\Local\Temp/ipykernel_8420/1061268490.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      all_params.lnValue[all_params['Params']=='Kburst']=np.log(all_params.value[all_params['Params']=="Kburst"])
    C:\Users\carte\AppData\Local\Temp/ipykernel_8420/1061268490.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      all_params.lnValue[all_params['Params']=='Klapse']=np.log(all_params.value[all_params['Params']=="Klapse"])
    C:\Users\carte\AppData\Local\Temp/ipykernel_8420/1061268490.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      all_params.lnValue[all_params['Params']=='P']=np.log((all_params.value[all_params['Params']=="P"]/(1-all_params.value[all_params['Params']=="P"])))
    C:\Users\carte\AppData\Local\Temp/ipykernel_8420/1061268490.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      all_params.lnValue[all_params['Params']=='q']=np.log((all_params.value[all_params['Params']=="q"]/(1-all_params.value[all_params['Params']=="q"])))
    


```python
display(all_params)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Params</th>
      <th>Interval</th>
      <th>Motivation</th>
      <th>Schedule</th>
      <th>variable</th>
      <th>value</th>
      <th>MouseID</th>
      <th>InitRsp</th>
      <th>lnValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>epsilon</td>
      <td>3</td>
      <td>B</td>
      <td>DRL</td>
      <td>LPM1</td>
      <td>27.227397</td>
      <td>1</td>
      <td>LPM</td>
      <td>3.304224</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c</td>
      <td>3</td>
      <td>B</td>
      <td>DRL</td>
      <td>LPM1</td>
      <td>0.109529</td>
      <td>1</td>
      <td>LPM</td>
      <td>-2.211564</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P</td>
      <td>3</td>
      <td>B</td>
      <td>DRL</td>
      <td>LPM1</td>
      <td>0.752748</td>
      <td>1</td>
      <td>LPM</td>
      <td>1.113320</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kburst</td>
      <td>3</td>
      <td>B</td>
      <td>DRL</td>
      <td>LPM1</td>
      <td>2.247934</td>
      <td>1</td>
      <td>LPM</td>
      <td>0.810012</td>
    </tr>
    <tr>
      <th>4</th>
      <td>q</td>
      <td>3</td>
      <td>B</td>
      <td>DRL</td>
      <td>LPM1</td>
      <td>0.244310</td>
      <td>1</td>
      <td>LPM</td>
      <td>-1.129195</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>c</td>
      <td>6</td>
      <td>PF</td>
      <td>FMI</td>
      <td>NPM10</td>
      <td>1.211060</td>
      <td>10</td>
      <td>NPM</td>
      <td>0.191496</td>
    </tr>
    <tr>
      <th>764</th>
      <td>P</td>
      <td>6</td>
      <td>PF</td>
      <td>FMI</td>
      <td>NPM10</td>
      <td>0.266228</td>
      <td>10</td>
      <td>NPM</td>
      <td>-1.013846</td>
    </tr>
    <tr>
      <th>765</th>
      <td>Kburst</td>
      <td>6</td>
      <td>PF</td>
      <td>FMI</td>
      <td>NPM10</td>
      <td>1.313623</td>
      <td>10</td>
      <td>NPM</td>
      <td>0.272789</td>
    </tr>
    <tr>
      <th>766</th>
      <td>q</td>
      <td>6</td>
      <td>PF</td>
      <td>FMI</td>
      <td>NPM10</td>
      <td>0.151613</td>
      <td>10</td>
      <td>NPM</td>
      <td>-1.722006</td>
    </tr>
    <tr>
      <th>767</th>
      <td>Klapse</td>
      <td>6</td>
      <td>PF</td>
      <td>FMI</td>
      <td>NPM10</td>
      <td>8.116420</td>
      <td>10</td>
      <td>NPM</td>
      <td>2.093889</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 9 columns</p>
</div>



```python
#Analysis of gamma parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig3 = plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
sns.barplot(x="Interval",y="lnValue",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="DRL") & (all_params['Params']=="epsilon")& (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('log(Epsilon)', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Nose Poke", fontsize=14)
plt.subplot(2,2,2)
sns.barplot(x="Interval",y="lnValue",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="DRL") & (all_params['Params']=="epsilon")& (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('log(Epsilon)', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Lever Press", fontsize=14)
plt.subplot(2,2,3)
sns.barplot(x="Interval",y="lnValue",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="FMI") & (all_params['Params']=="epsilon")& (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('log(Epsilon)', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Nose Poke", fontsize=14)
plt.subplot(2,2,4)
sns.barplot(x="Interval",y="lnValue",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="FMI") & (all_params['Params']=="epsilon")& (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('log(Epsilon)', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Lever Press", fontsize=14)
plt.tight_layout()
```


    
![png](output_30_0.png)
    



```python
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig3 = plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="DRL") & (all_params['Params']=="c")& (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('c', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Nose Poke", fontsize=14)
plt.subplot(2,2,2)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="DRL") & (all_params['Params']=="c")& (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('c', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Lever Press", fontsize=14)
plt.subplot(2,2,3)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="FMI") & (all_params['Params']=="c")& (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('c', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Nose Poke", fontsize=14)
plt.subplot(2,2,4)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="FMI") & (all_params['Params']=="c")& (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('c', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Lever Press", fontsize=14)
plt.tight_layout()
```


    
![png](output_31_0.png)
    



```python
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig3 = plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="DRL") & (all_params['Params']=="P")& (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('P', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Nose Poke", fontsize=14)
plt.subplot(2,2,2)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="DRL") & (all_params['Params']=="P")& (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('P', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Lever Press", fontsize=14)
plt.subplot(2,2,3)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="FMI") & (all_params['Params']=="P")& (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('P', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Nose Poke", fontsize=14)
plt.subplot(2,2,4)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="FMI") & (all_params['Params']=="P")& (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('P', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Lever Press", fontsize=14)
plt.tight_layout()
```


    
![png](output_32_0.png)
    



```python
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig3 = plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="DRL") & (all_params['Params']=="q")& (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('q', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Nose Poke", fontsize=14)
plt.subplot(1,2,2)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="DRL") & (all_params['Params']=="q")& (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('q', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Lever Press", fontsize=14)
plt.tight_layout()
```


    
![png](output_33_0.png)
    



```python
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig3 = plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="DRL") & (all_params['Params']=="Kburst")& (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('Kburst', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Nose Poke", fontsize=14)
plt.subplot(2,2,2)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="DRL") & (all_params['Params']=="Kburst")& (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('Kburst', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Lever Press", fontsize=14)
plt.subplot(2,2,3)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="FMI") & (all_params['Params']=="Kburst")& (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('Kburst', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Nose Poke", fontsize=14)
plt.subplot(2,2,4)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="FMI") & (all_params['Params']=="Kburst")& (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('Kburst', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("FMI: Lever Press", fontsize=14)
plt.tight_layout()
```


    
![png](output_34_0.png)
    



```python
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig3 = plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="DRL") & (all_params['Params']=="Klapse")& (all_derivedstats['InitRsp']=="NPM"))],estimator=np.mean)
plt.ylabel('Klapse', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Nose Poke", fontsize=14)
plt.subplot(1,2,2)
sns.barplot(x="Interval",y="value",hue="Motivation",ci=68,data=all_params[((all_params['Schedule']=="DRL") & (all_params['Params']=="Klapse")& (all_derivedstats['InitRsp']=="LPM"))],estimator=np.mean)
plt.ylabel('Klapse', fontsize=14)
plt.xlabel('Interval (s)', fontsize=14)
plt.title("DRL: Lever Press", fontsize=14)
plt.tight_layout()
```


    
![png](output_35_0.png)
    



```python
all_params.to_csv("topography_params.csv")

```

The output of the analysis conducted in R can be found as topography_params.Rmd and .html in the Github repo. Here I summarize the main findings. Note that because mice were trained in sequence from DRL to FMI  

1. No effects on Epsilon
2. c is sensitive to interval and initiating response
3. P is sensitive to motivation and schedule
4. K burst is sensitive to interval*Schedule, interval*init, schedule*init
5. DRL only: q is sensitive to interval*motivation*init
6. DRL only: K lapse is sensitive to interval (decreases with interval, schedule strain?) 

***REWRITE WITH NARRATIVE and log-BF10***


```python

```
