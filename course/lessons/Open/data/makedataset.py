#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:37:53 2021

@author: davsu428
"""


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png')
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight

actors=pd.read_csv('Pudding-Film-Dialogue-Clean.csv')

df = pd.DataFrame(columns={'Number of female actors','Number of male actors','Number of words lead 1','Number of words lead 2','Total words'
                           ,'Number words male','Number words female','Gross','Age men','Age women','Year','Lead female','Lead male','Title'})

current_title=actors.iloc[0]['title']
num_female=0
num_male=0
num_words_lead_female=0
num_words_lead_male=0
total_words=0
words_male=0
words_female=0
age_men =[]
age_women =[]
lead_female=''
lead_male=''


np.random.seed(1)

    
for i,actor in actors.iterrows():

    if (current_title==actor['title']):
        if (actor['gender']=='woman'):
            num_female=num_female+1
            num_words=actor['words']
            words_female=words_female+num_words
            if not(np.isnan(actor['age'])):
                age_women = age_women + [actor['age']]
            if num_words>num_words_lead_female:
                num_words_lead_female=num_words
                lead_female=actor['character']
                
        elif (actor['gender']=='man'):
            num_male=num_male+1
            num_words=actor['words']
            words_male=words_male+num_words
            if not(np.isnan(actor['age'])):
                age_men = age_men + [actor['age']]
            if num_words>num_words_lead_male:
                num_words_lead_male=num_words
                lead_male=actor['character']
        else:
            print('no gender' + actor['character'])
    
        gross=actor['gross']
        year=actor['release_year']
        

        
        gross=actor['gross']
        year=actor['release_year']
        
    
    else:
        #Randomise lead
        
        if np.random.random()<1/2:
            lead='Female'
            num_words_lead_1=num_words_lead_female
            num_words_lead_2=num_words_lead_male
        else:
            lead='Male'
            num_words_lead_1=num_words_lead_male
            num_words_lead_2=num_words_lead_female
               
        newrow=pd.Series({'Number of female actors':num_female, 'Number of male actors':num_male,
                            'Number of words lead 1':num_words_lead_1,
                            'Number of words lead 2':num_words_lead_2,
                            'Total words':words_female+words_male,
                            'Number words male':words_male,
                            'Number words female':words_female,
                            'Gross':gross ,
                            'Age men': age_men,
                            'Age women': age_women,
                            'Year':year,
                            'Lead female':lead_female,
                            'Lead male':lead_male,
                            'Title':current_title,
                            'Lead 1':lead}
    
    )
        df=df.append(newrow,ignore_index=True)
        
        
        #reset all the variables.
        current_title=actor['title']
        num_female=0
        num_male=0
        num_words_lead_female=0
        num_words_lead_male=0
        total_words=0
        words_male=0
        words_female=0
        age_men =[]
        age_women =[]
        lead_female=''
        lead_male=''
        
df['Mean Age Male'] = (df.iloc[:]['Age men']).apply(np.mean)
df['Mean Age Female'] = (df.iloc[:]['Age women']).apply(np.mean)
#df['Std Age Male'] = (df.iloc[:]['Age men']).apply(np.std)
#df['Std Age Female'] = (df.iloc[:]['Age women']).apply(np.std)
        
df=df.dropna()
# sampling indices for training


trainI = np.random.choice(df.shape[0], size=1500, replace=False)
trainIndex = df.index.isin(trainI)
train = df.iloc[trainIndex]  # training set
test = df.iloc[~trainIndex]  # test set
        
model = skl_lm.LogisticRegression(solver='lbfgs')

train_variables=['Number words female', 'Total words', 'Number of words lead 1',
       'Number of male actors', 'Year','Number of female actors', 'Number of words lead 2', 
       'Number words male', 'Gross']

train.to_csv('train_all.csv', index=False)
test.to_csv('test_all.csv', index=False)

train_student = train[train_variables + ['Lead 1']]
test_student = test[train_variables]

train_student.to_csv('train.csv', index=False)
test_student.to_csv('test.csv', index=False)


