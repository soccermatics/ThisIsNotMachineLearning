"""
Election predictions
====================

fivethirtyeight

"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 12/2.54, 12/2.54


predictions=pd.read_csv('predictions.csv',delimiter=';')

DemProb_market=predictions['Market']
DemProb_538=predictions['FiveThirtyEight']
DemWin=predictions['Result']

steps=np.array([0,0.05,0.25,0.5,0.75,0.95,1.01])
x=np.array([0.02,0.15,0.375,0.625,0.85,0.99])
xtlabel=['0-5','5-25','25-50','50-75','75-95','95-100']

n1=np.zeros(len(steps)-1)
n1_size=np.zeros(len(steps)-1)
n2=np.zeros(len(steps)-1)
n2_size=np.zeros(len(steps)-1)


for DemProb in [DemProb_market,DemProb_538]:
     
    for j,step in enumerate(steps[:-1]):
    
        inds=np.where((DemProb / 100 >= step) & (DemProb / 100 < steps[j + 1]))
        n1[j]=np.sum(DemWin[inds[0]]) / len(inds[0])
        n1_size[j]=len(inds[0])*10

    #Plot predictions vs. outcomes
    fig,ax=plt.subplots(1)  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(-0.05,1.05) 
    ax.set_xticks(x)   
    ax.set_xticklabels(xtlabel)       
    ax.scatter(x, n1,s=n1_size)
    ax.plot([-1,2],[-1,2],':')
    ax.set_title(DemProb.name)
    ax.set_xlabel('Predicted probability of Democrat win (%)')
    ax.set_ylabel('Proportion of states won by Democrats (%)')
    plt.show()


    # Brier score
    Brier_score=np.mean(np.power(DemProb/100 - DemWin,2))

    print('Brier score for %s is %.4f'%(DemProb.name,Brier_score))
