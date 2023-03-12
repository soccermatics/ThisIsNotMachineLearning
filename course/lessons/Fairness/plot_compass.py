"""
Propublica Analysis
===================

In May 2016, a ProPublica article <a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing/">(Angwin et al. 2016)</a> 
analysed the COMPAS Recidivism Algorithm. Kenny Kyunghoon Lee, `adpated the original ProPublica analysis for Python <https://github.com/kennylee15/blog.git>`_.

* <a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing/">Read the ProPublica story</a>.
* <a href="https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm">Read the technical note</a>.


COMPAS score for recidivism
---------------------------

First we load the data. We select fields for severity of charge, number of priors, demographics, age, sex, compas scores, and whether each person was accused of a crime within two years. Then we pront the table.
"""

import pandas as pd
import numpy as np
from pylab import rcParams
rcParams['figure.figsize'] = 12/2.54, 12/2.54

pd.options.mode.chained_assignment = None  # default='warn'

dataURL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
dfRaw = pd.read_csv(dataURL)

dfFiltered = (dfRaw[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 
             'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score', 
             'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
             .loc[(dfRaw['days_b_screening_arrest'] <= 30) & (dfRaw['days_b_screening_arrest'] >= -30), :]
             .loc[dfRaw['is_recid'] != -1, :]
             .loc[dfRaw['c_charge_degree'] != 'O', :]
             .loc[dfRaw['score_text'] != 'N/A', :]
             )
dfFiltered = dfFiltered.rename(columns={"race": "Group"})

print('Number of rows: {}'.format(len(dfFiltered.index)))


###########################################################################
# Score distribution by group
# --------------------------
# The distribution of decile scores, which are often presented to judges alongside risk classification (High, Medium and Low), suggests disparity. There is no clear 
# downtrend in decile scores for African American defendents, unlike for the Caucasian counterpart.  
#
# COMPAS scores for each defendant ranged from 1 to 10, with ten being the highest risk. Scores 1 to 4 were labeled by COMPAS as “Low”; 5 to 7 were labeled “Medium”; and # 8 to 10 were labeled “High.”
# We make a simple cross tabulation of score categories by Group.

pd.crosstab(dfFiltered['score_text'],dfFiltered['Group'])

###########################################################################
# Decile scores that correspond to score categories

pd.crosstab(dfFiltered['score_text'],dfFiltered['decile_score'])


###########################################################################
# Histograms of decile scores

import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 12/2.54, 12/2.54


ax=sns.countplot(x='decile_score', hue='Group', data=dfFiltered.loc[
                (dfFiltered['Group'] == 'African-American') | (dfFiltered['Group'] == 'Caucasian'),:
            ])
    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Distribution of Decile Scores")
plt.xlabel('Decile Score')
plt.ylabel('Count')
plt.show()

###########################################################################
# Logistic regression
# -------------------
# The cateogorical data is transformed into dummy variables and logistic regressions are run that consider Group, age, criminal history, future recidivism, charge degree, # gender and age. 'High' and 'Medium' categories are combined following the ProPublica analysis.

 
import statsmodels.api as sm
from statsmodels.formula.api import logit
catCols = ['score_text','age_cat','sex','Group','c_charge_degree']
dfFiltered.loc[:,catCols] = dfFiltered.loc[:,catCols].astype('category')

# dfDummies = pd.get_dummies(data = dfFiltered.loc[dfFiltered['score_text'] != 'Low',:], columns=catCols)
dfDummies = pd.get_dummies(data = dfFiltered, columns=catCols)

# Clean column names
new_column_names = [col.lstrip().rstrip().lower().replace(" ", "_").replace("-", "_") for col in dfDummies.columns]
dfDummies.columns = new_column_names

# We want another variable that combines Medium and High
dfDummies['score_text_medhi'] = dfDummies['score_text_medium'] + dfDummies['score_text_high']

# R-style specification
formula = 'score_text_medhi ~ sex_female + age_cat_greater_than_45 + age_cat_less_than_25 + group_african_american + group_asian + group_hispanic + group_native_american + group_other + priors_count + c_charge_degree_m + two_year_recid'

score_mod = logit(formula, data = dfDummies).fit()
print(score_mod.summary())

# Black defendants were 45.3 percent more likely than white defendants to receive a higher score, correcting for prior crimes, future criminality.
control = np.exp(-1.5255) / (1 + np.exp(-1.5255))
np.exp(0.4772) / (1 - control + (control * np.exp(0.4772)))

# Female defendants are 19.4% more likely than men to get a higher score.


np.exp(0.2213) / (1 - control + (control * np.exp(0.2213)))


###########################################################################
# Risk of "violent" recidivism
# ----------------------------
#
# ProPublica authors followed the FBI’s definition of violent crime, a category that includes murder, manslaughter, forcible rape, robbery and aggravated assault.

dataURL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years-violent.csv'
dfRaw = pd.read_csv(dataURL)

dfFiltered = (dfRaw[['age', 'c_charge_degree', 'race', 'age_cat', 'v_score_text', 
             'sex', 'priors_count', 'days_b_screening_arrest', 'v_decile_score', 
             'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
             .loc[(dfRaw['days_b_screening_arrest'] <= 30) & (dfRaw['days_b_screening_arrest'] >= -30), :]
             .loc[dfRaw['is_recid'] != -1, :]
             .loc[dfRaw['c_charge_degree'] != 'O', :]
             .loc[dfRaw['v_score_text'] != 'N/A', :]
             )

dfFiltered = dfFiltered.rename(columns={"race": "Group"})

print('Number of rows: {}'.format(len(dfFiltered.index)))

ax=sns.countplot(x='v_decile_score', hue='Group', data=dfFiltered.loc[
                (dfFiltered['Group'] == 'African-American') | (dfFiltered['Group'] == 'Caucasian'),:
            ])
    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Distribution of Violent Decile Scores")
plt.xlabel('Decile Score')
plt.ylabel('Count')
plt.show()

###########################################################################
#  COMPAS violent risk scores also show a disparity in distribution between white and black defendants.

catCols = ['v_score_text','age_cat','sex','Group','c_charge_degree']
dfFiltered.loc[:,catCols] = dfFiltered.loc[:,catCols].astype('category')

dfDummies = pd.get_dummies(data = dfFiltered, columns=catCols)

# Clean column names
new_column_names = [col.lstrip().rstrip().lower().replace (" ", "_").replace ("-", "_") for col in dfDummies.columns]
dfDummies.columns = new_column_names

# We want another variable that combines Medium and High
dfDummies['v_score_text_medhi'] = dfDummies['v_score_text_medium'] + dfDummies['v_score_text_high']

###########################################################################
# Regression analysis - logistic regression

formula = 'v_score_text_medhi ~ sex_female + age_cat_greater_than_45 + age_cat_less_than_25 + group_african_american + group_asian + group_hispanic + group_native_american + group_other  + priors_count + c_charge_degree_m + two_year_recid'

score_mod = logit(formula, data = dfDummies).fit()
print(score_mod.summary())

# Black defendants were 77.4 percent more likely than white defendants to receive a higher score, correcting for prior crimes, future criminality.

control = np.exp(-2.2427) / (1 + np.exp(-2.2427))
np.exp(0.6589) / (1 - control + (control * np.exp(0.6589)))



###########################################################################
# Age distribution
# ----------------

dfAA = dfFiltered[(dfFiltered['Group']=='African-American')]
dfCA = dfFiltered[(dfFiltered['Group']=='Caucasian')]


def FormatFigure(ax):
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('')
    ax.set_xlabel('Age')
    ax.set_ylabel('Proportion of people')
    ax.set_xticks(np.arange(18+2.5,78,step=5)) 
    ax.set_xticklabels(np.arange(18,78,step=5))
    ax.set_yticks(np.arange(0,0.06,step=0.01))
    ax.set_ylim(0,0.055)
    
    
fig,(ax1,ax2)=plt.subplots(2,1)
ax1.hist(dfAA['age'], np.arange(18,78,1), color='orange', edgecolor = 'black',linestyle='-',density='True',alpha=0.5, align='right')
FormatFigure(ax1)
ax2.hist(dfCA['age'], np.arange(18,78,1), color='orange', edgecolor = 'black',linestyle='-',density='True',alpha=0.5, align='right')
FormatFigure(ax2)

plt.show()


###########################################################################
# What to add
# -----------
#
# That the whole predictor is basically age and previos convictions
# 
# Show with regression.



###########################################################################
# Acknowledgement
# --------------
#
# The text and code in this page is modfied from Kenny Kyunghoon Lee, `who adpated the original ProPublica
# analysis for Python <https://github.com/kennylee15/blog.git>`_. We are grateful for him making this available under `licence <https://github.com/kennylee15/blog/blob/gh-pages/LICENSE.txt>`_.  