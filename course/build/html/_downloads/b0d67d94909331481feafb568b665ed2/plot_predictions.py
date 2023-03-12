"""
.. _predictions2022:

Predictions World Cup 2022
==========================

How it went in 2018
-------------------

Marius, Jan and myself placed a series of bets on the Wrold Cup 2018 and we won.
Later I wrote...

.. image:: ../../images/lesson1/TheResult.png
   :width: 640
   :align: center

I am always in two minds as to whether to offer gambling tips. In the 
Ten Equations I follow Jan and Marius as they start to profit from their gambling
enterprise. They adopted a scientific approach, while many people looking for
quick tips do not. This proved to be a common theme when I looked at the role of
maths in society: those with the knowledge accumulate financial resources, 
those without the appropriate training lose out. 

Nevertheless, it isn't an interesting experiment if we don't make predictions. 
So here we go...

Try it in 2022
--------------

We now load in the odds for World Cup 2022 and try to find an edge using the model. 
When I collected the odds used here, they were still close to the opening level. 
So I use the paramters measured for that value. You can upload closing odds 
before the match and change the alpha or beta values accordingly.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

alpha = 1.0372197647675445
beta = 1.1051122982366652

odds_load = pd.read_csv("../data/WC2022.csv", delimiter=';')

totprob=1/odds_load['Home']+1/odds_load['Draw']+1/odds_load['Away']
odds_load = odds_load.assign(homeprob=1/odds_load['Home']/totprob)
odds_load = odds_load.assign(awayprob=1/odds_load['Away']/totprob)
odds_load = odds_load.assign(drawprob=1/odds_load['Draw']/totprob) 
    
    
# Calculate in terms of favourite.
odds_load = odds_load.assign(favodds=np.minimum(odds_load['Home'],odds_load['Away']))
odds_load = odds_load.assign(underdogodds=np.maximum(odds_load['Home'],odds_load['Away']))
odds_load = odds_load.assign(favprob=np.maximum(odds_load['homeprob'],odds_load['awayprob']))
odds_load = odds_load.assign(favfair=1/odds_load['favprob'])


odds_load['favourite'] = ''
odds_load['underdog'] = ''

for i,odds_row in odds_load.iterrows():
    if (odds_row['homeprob'] > odds_row['awayprob']):
        odds_load.at[i,'favourite'] = odds_row['Home Team']
        odds_load.at[i,'underdog'] = odds_row['Away Team']
    else:
        odds_load.at[i,'favourite'] = odds_row['Away Team']
        odds_load.at[i,'underdog'] = odds_row['Home Team']  
        
odds_load = odds_load.assign(favnewprob=1/(1+alpha*np.power(odds_load['favfair']-1,beta)))
odds_load = odds_load.assign(underdognewprob=1-odds_load['favnewprob']-odds_load['drawprob'])
odds_load = odds_load.assign(favfairodds=1/odds_load['favnewprob'])
odds_load = odds_load.assign(underdogfairodds=1/odds_load['underdognewprob'])

for i,odds_row in odds_load.iterrows():
    if (odds_row['favfairodds']<odds_row['favodds']):
        print('Back favourite %s on odds better than %.2f.' % (odds_row['favourite'],odds_row['favfairodds']) )
    elif (odds_row['underdogfairodds']<odds_row['underdogodds']):
        print('Back underdog %s on odds better than %.2f.' % ( odds_row['underdog'], odds_row['underdogfairodds']) )
    #else:
    #    print('No bet %s vs. %s' % (odds_row['favourite'],odds_row['underdog']))

#####################################################################
# 
# Please gamble responsibly.