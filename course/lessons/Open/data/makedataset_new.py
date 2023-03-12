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

df = pd.DataFrame(columns={'Number of female actors','Number of male actors','Difference in words lead and co-lead','Total words'
                           ,'Number words male','Number words female','Gross','Age men','Age women','Year','Lead female','Lead male','Title','Age Lead', 'Age Co-Lead'})

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

    if actor['age']>200:
        actor['age']=np.nan
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
                lead_female_age=actor['age']
                
        elif (actor['gender']=='man'):
            num_male=num_male+1
            num_words=actor['words']
            words_male=words_male+num_words
            if not(np.isnan(actor['age'])):
                age_men = age_men + [actor['age']]
            if num_words>num_words_lead_male:
                num_words_lead_male=num_words
                lead_male=actor['character']
                lead_male_age=actor['age']
        else:
            print('no gender' + actor['character'])
    
        gross=actor['gross']
        year=actor['release_year']
        
        gross=actor['gross']
        year=actor['release_year']
        
    
    else:
        
        total_words=words_male+words_female
        if (num_words_lead_female>=num_words_lead_male):
            lead='Female'
            number_of_words_difference=num_words_lead_female-num_words_lead_male
            number_of_words_lead=num_words_lead_female
            words_female=words_female-num_words_lead_female
            lead_age=lead_female_age
            colead_age=lead_male_age
        else:
            lead='Male'
            number_of_words_difference=num_words_lead_male-num_words_lead_female
            number_of_words_lead=num_words_lead_male
            words_male=words_male-num_words_lead_male
            lead_age=lead_male_age
            colead_age=lead_female_age
               
        newrow=pd.Series({'Number of female actors':num_female, 'Number of male actors':num_male,
                            'Difference in words lead and co-lead':number_of_words_difference,
                            'Total words':total_words,
                            'Number words male':words_male,
                            'Number words female':words_female,
                            'Number of words lead': number_of_words_lead,
                            'Gross':gross ,
                            'Age men': age_men,
                            'Age women': age_women,
                            'Year':year,
                            'Lead female':lead_female,
                            'Lead male':lead_male,
                            'Title':current_title,
                            'Age Lead': lead_age,
                            'Age Co-Lead': colead_age,
                            'Lead':lead}
    
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


trainI = np.random.choice(df.shape[0], size=1400, replace=False)
trainIndex = df.index.isin(trainI)
train = df.iloc[trainIndex]  # training set
test = df.iloc[~trainIndex]  # test set
        
model = skl_lm.LogisticRegression(solver='lbfgs')

train_variables=['Number words female', 'Total words','Number of words lead', 'Difference in words lead and co-lead',
       'Number of male actors', 'Year','Number of female actors', 
       'Number words male', 'Gross','Mean Age Male','Mean Age Female', 'Age Lead', 'Age Co-Lead']

train.to_csv('train_all.csv', index=False)
test.to_csv('test_all.csv', index=False)

train_student = train[train_variables + ['Lead']]
test_student = test[train_variables]

train_student.to_csv('train.csv', index=False)
test_student.to_csv('test.csv', index=False)


