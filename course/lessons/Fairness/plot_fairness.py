"""
Measuring fairness
==================

Imagine your colleagues have created a supervised machine learning model
to find people who might be interested in studying at a university in
Sweden, based on their activity on a social networking site. Their
algorithm either recommends or doesn’t recommend the course to users.
They have tested it on two different groups of people (600 non-Swedes
and 1 200 Swedes), all of whom would be eligible for the course and have
given permission for their data to be used. As a test, your colleagues
first applied the method, then asked the potential students whether or
not they would be interested in the course. To illustrate their results,
they produced the confusion matrices below for non-Swedes and Swedes.

.. container::
   :name: tab:SwNonSw

   .. table:: Proportion of people shown and/or interested in a course
   for an imagined machine learning algorithm. The top table is for
   non-Swedes (in this case we can think of them as citizens of another
   country, but who are eligible to study in Sweden); the bottom table
   is for Swedes.

      +------------------------------+------------------+------------------+
      |                              | Not Interested   | Interested       |
      +------------------------------+------------------+------------------+
      | **Non-Swedes**               | (:math:`y=-1`)   | (:math:`y=1`)    |
      +------------------------------+------------------+------------------+
      | Not recommended course       | TN = :math:`300` | FN = :math:`100` |
      | (:math:`\yhat(\bx) = -1`)    |                  |                  |
      +------------------------------+------------------+------------------+
      | Recommended course           | FP = :math:`100` | TP = :math:`100` |
      | (:math:`\yhat(\bx) = 1`)     |                  |                  |
      +------------------------------+------------------+------------------+
      |                              |                  |                  |
      +------------------------------+------------------+------------------+
      |                              | Not Interested   | Interested       |
      +------------------------------+------------------+------------------+
      | **Swedes**                   | (:math:`y=-1`)   | (:math:`y=1`)    |
      +------------------------------+------------------+------------------+
      | Not recommended course       | TN = :math:`400` | FN = :math:`50`  |
      | (:math:`\yhat(\bx) = -1`)    |                  |                  |
      +------------------------------+------------------+------------------+
      | Recommended course           | FP = :math:`350` | TP = :math:`400` |
      | (:math:`\yhat(\bx) = 1`)     |                  |                  |
      +------------------------------+------------------+------------------+

Let's create these confusion matrices in 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def drawTable(Conf,group_name):
        FP =Conf[1][0]
        TP =Conf[1][1]
        P = Conf[1][1] + Conf[0][1] #Positive class
        N = Conf[0][0] + Conf[1][0] #Negative class

        print("----------------------------")
        print(group_name)
        print("----------------------------")
        print(f"TN = {N-FP:.0f} | FN = {P - TP:.0f} ")
        print(f"FP = {FP:.0f}| TP = {TP:.0f} ")
        print("----------------------------\n")
        print(f"False positive rate: {FP/N:.2f}")
        print(f"True positive rate: {TP/P:.2f}")
        print(f"Precision: {TP/(TP+FP):.2f}")
        print(f"Misclassification error: {1-(TP + (N-FP))/(N+P):.2f}\n\n")
    

#non-Swedes
ConNSw = np.array([[300,100],[100,100]])
#Swedes
ConSw = np.array([[400,50],[350,400]])


drawTable(ConNSw,"Non-Swedes")
drawTable(ConSw,"Swedes")


##############################################################################
# Let’s focus on the question of whether the algorithm performs equally
# well on both groups, non-Swedes and Swedes. We might call this property
# ‘fairness’. Does the method treat the two groups fairly? To answer this
# question, we first need to quantify fairness. One suggestion here would
# be ask if the method performs equally well for both groups. Referring to
# `[tab:confmatterm] <#tab:confmatterm>`__, and
# `[ch:evaluation] <#ch:evaluation>`__ in general, we see that one way of
# measuring performance is to use misclassification error. For
# `1.1 <#tab:SwNonSw>`__, the misclassification error is
# :math:`(100+100)/600=1/3` for non-Swedes and :math:`(50+350)/1\,200=1/3`
# for Swedes. It has the same performance for both categories.

# It is now that alarm bells should start to ring about equating fairness
# with performance. If we look at the false negatives (FN) for both cases,
# we see that there are twice as many non-Swede FN cases as Swedish cases
# (100 vs. 50), despite their being twice as many Swedes as non-Swedes.
# This can be made more precise by calculating the false negative rate (or
# miss rate), i.e. FN/(TP+FN). This is
# :math:`100/(100+100)=1/2` for non-Swedes and :math:`50/(400+50)=1/9` for
# Swedes. This new result can be put in context by noting that Swedes have
# a slightly greater tendency to be interested in the course (450 out of 
# 1200 vs. 200 out of 600). However, an interested non-Swede is 4.5 times
# more likely *not* to be recommended the course than an interested Swede.
# A much larger difference than that observed in the original data.

# There are other fairness calculations we can do. Imagine we are
# concerned with intrusive advertising, where people are shown adverts
# that are uninteresting for them. The probability of experiencing a
# recommendation that is uninteresting is the false positive rate,
# FP/(TN+FP). This is :math:`100/(300+100)=1/4` for non-Swedes and
# :math:`350/(350+400)=7/15` for Swedes. Swedes receive almost twice as
# many unwanted recommendations as non-Swedes. Now it is the Swedes who
# are discriminated against!

# This is a fictitious example, but it serves to illustrate the first
# point we now want to make: *There is no single function for measuring
# fairness*. In some applications, fairness is perceived as
# misclassification; in others it is false negative rates, and in others
# it is expressed in terms of false positives. It depends strongly on the
# application. If the data above had been for a criminal sentencing
# application, where ‘positives’ are sentenced to longer jail terms, then
# problems with the false positive rate would have serious consequences
# for those sentenced on the basis of it. If it was for a medical test,
# where those individuals not picked up by the test had a high probability
# of dying, then the false negative rate is most important for judging
# fairness.

# As a machine learning engineer, you should never tell a client that your
# algorithm is fair. You should instead explain how your model performs in
# various aspects related to their conception of fairness. This insight is
# well captured by Dwork and colleagues’ article, ‘Fairness Through
# Awareness’ :cite:p:`dwork2012fairness`, which is
# recommended further reading. Being fair is about being aware of the
# decisions we make both in the design and in reporting the outcome of our
# model.

# .. _`sec:nofairness`:

# Complete Fairness Is Mathematically Impossible
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# We now come to an even more subtle point: *It is mathematically
# impossible to create models that fulfil all desirable fairness
# criteria*. Let’s demonstrate this point with another example, this time
# using a real application. The Compas algorithm was developed by a
# private company, Northpointe, to help with criminal sentencing
# decisions. The model used logistic regression with input variables
# including age at first arrest, years of education, and questionnaire
# answers about family background, drug use, and other factors to predict
# an output variable as to whether the person would reoffend
# :raw-latex:`\parencite{sumpter2018outnumbered}`. Race was not included
# in the model. Nonetheless, when tested – as part of a a study by Julia
# Angwin and colleagues at Pro-Publica
# :raw-latex:`\parencite{larson2016we}` – on an independently collected
# data set, the model gave different predictions for black defendants than
# for white. The results are shown in the form of a confusion matrix in
# Table `1.2 <#ch12:tab12.2>`__, for re-offending over the next two years.

# .. container::
#    :name: ch12:tab12.2

#    .. table:: Confusion matrix for the Pro-Publica study of the Compas
#    algorithm. For details see :raw-latex:`\textcite{larson2016we}`.

#       +----------------------+----------------------+----------------------+
#       | **Black defendants** | Didn’t reoffend      | Reoffended           |
#       |                      | (:math:`y=-1`)       | (:math:`y=1`)        |
#       +======================+======================+======================+
#       | Lower risk           | TN = :math:`990`     | FN = :math:`532`     |
#       | (:mat                |                      |                      |
#       | h:`\yhat(\bx) = -1`) |                      |                      |
#       +----------------------+----------------------+----------------------+
#       | Higher risk          | FP = :math:`805`     | TP = :math:`1\,369`  |
#       | (:ma                 |                      |                      |
#       | th:`\yhat(\bx) = 1`) |                      |                      |
#       +----------------------+----------------------+----------------------+
#       |                      |                      |                      |
#       +----------------------+----------------------+----------------------+
#       | **White defendants** | Didn’t reoffend      | Reoffended           |
#       |                      | (:math:`y=-1`)       | (:math:`y=1`)        |
#       +----------------------+----------------------+----------------------+
#       | Lower risk           | TN = :math:`1\,139`  | FN = :math:`461`     |
#       | (:mat                |                      |                      |
#       | h:`\yhat(\bx) = -1`) |                      |                      |
#       +----------------------+----------------------+----------------------+
#       | Higher risk          | FP = :math:`349`     | TP = :math:`505`     |
#       | (:ma                 |                      |                      |
#       | th:`\yhat(\bx) = 1`) |                      |                      |
#       +----------------------+----------------------+----------------------+

# Angwin and her colleagues pointed out that the false positive rate for
# black defendants, :math:`805/(990+805)=44.8`\ %, is almost double that
# of white defendants, :math:`349/(349+1\,139)=23.4`\ %. This difference
# cannot be accounted for simply by overall reoffending rates: although
# this is higher for black defendants (at 51.4% arrested for another
# offence within two years), when compared to white defendants (39.2%),
# these differences are smaller than the differences in false positive
# rates. On this basis, the model is clearly unfair. The model is also
# unfair in terms of true positive rate (recall). For black defendants,
# this is :math:`1\,369/(532+1369)=72.0`\ % versus
# :math:`505/(505+461)=52.2`\ % for white defendants. White offenders who
# go on to commit crimes are more likely to be classified as lower risk.

# In response to criticism about the fairness of their method, the company
# Northpointe countered that in terms of performance, the precision
# (positive predictive value) was roughly equal for both groups:
# :math:`1\,369/(805+1369)=63.0`\ % for black defendants and
# :math:`505/(505+349)=59.1`\ % for white
# :raw-latex:`\parencite{sumpter2018outnumbered}`. In this sense the model
# is fair, in that it has the same performance for both groups. Moreover,
# Northpointe argued that it is precision which is required, by law, to be
# equal for different categories. Again this is the problem we highlighted
# above, but now with serious repercussions for the people this algorithm
# is applied to: black people who won’t later reoffend are more likely to
# classified as high risk than white people.

# Would it be possible (in theory) to create a model that was fair in
# terms of both false positives and precision? To answer this question,
# consider the confusion matrix in Table `1.3 <#ch12:tab12.3>`__.

# .. container::
#    :name: ch12:tab12.3

#    .. table:: Generic confusion matrix.

#       +----------------------+----------------------+----------------------+
#       | **Category 1**       | Negative             | Positive :math:`y=1` |
#       |                      | :math:`y=-1`         |                      |
#       +======================+======================+======================+
#       | Predicted negative   | :math:`n_1-f_1`      | :math:`p_1-t_1`      |
#       | (:mat                |                      |                      |
#       | h:`\yhat(\bx) = -1`) |                      |                      |
#       +----------------------+----------------------+----------------------+
#       | Predicted positive   | :math:`f_1`          | :math:`t_1`          |
#       | (:ma                 |                      |                      |
#       | th:`\yhat(\bx) = 1`) |                      |                      |
#       +----------------------+----------------------+----------------------+
#       |                      |                      |                      |
#       +----------------------+----------------------+----------------------+
#       | **Category 2**       | Negative             | Positive :math:`y=1` |
#       |                      | :math:`y=-1`         |                      |
#       +----------------------+----------------------+----------------------+
#       | Predicted negative   | :math:`n_2-f_2`      | :math:`p_2-t_2`      |
#       | (:mat                |                      |                      |
#       | h:`\yhat(\bx) = -1`) |                      |                      |
#       +----------------------+----------------------+----------------------+
#       | Predicted positive   | :math:`f_2`          | :math:`t_2`          |
#       | (:ma                 |                      |                      |
#       | th:`\yhat(\bx) = 1`) |                      |                      |
#       +----------------------+----------------------+----------------------+

# Here, :math:`n_i` and :math:`p_i` are the number of individuals in the
# negative and positive classes, and :math:`f_i` and :math:`t_i` are the
# number of false and true positives, respectively. The values of
# :math:`n_i` and :math:`p_i` are beyond the modeller’s control; they are
# determined by outcomes in the real world (does a person develop cancer,
# commit a crime, etc.). The values :math:`f_i` and :math:`t_i` are
# determined by the machine learning algorithm. For each category 1, we
# are constrained by a tradeoff between :math:`f_1` and :math:`t_1`, i.e.
# as determined by the ROC for model 1. A similar constraint applies to
# category 2. We can’t make our model arbitrarily accurate.

# However, we can (potentially using the ROC for each category as a guide)
# attempt to tune :math:`f_1` and :math:`f_2` independently of each other.
# In particular, we can ask that our model has the same false positive
# rate for both categories, i.e. :math:`f_1/n_1=f_2/n_2`, or

# .. math:: f_1 = \frac{n_1 f_2}{n_2}. \label{eq:practice:fequality}

# In practice, such a balance may be difficult to achieve, but our purpose
# here is to show that limitations exist even when we can tune our model
# in this way. Similarly, let’s assume we can specify that the model has
# the same true positive rate (recall) for both categories,

# .. math:: t_1= \frac{p_1 t_2}{p_2}. \label{eq:practice:tequality}

# Equal precision of the model for both categories is determined by
# :math:`t_1/(t_1+f_1)=t_2/(t_2+f_2)`. Substituting
# `[eq:practice:fequality] <#eq:practice:fequality>`__ and
# `[eq:practice:tequality] <#eq:practice:tequality>`__ in to this equality
# gives

# .. math:: \frac{ t_2}{t_2+ \frac{p_2 n_1 f_2}{p_1 n_2}}=\frac{t_2}{t_2+f_2},

# which holds only if :math:`f_1=f_2=0` or if

# .. math:: \frac{p_1}{n_1}=\frac{p_2}{n_2}. \label{eq:practice:precisionequality}

# In words, Equation
# `[eq:practice:precisionequality] <#eq:practice:precisionequality>`__
# implies that we can only achieve equal precision when the classifier is
# perfect on the positive class or when the ratios of positive numbers of
# people in the positive and negative classes for both categories are
# equal. Both of these conditions are beyond our control as modellers. In
# particular, the number in each class for each category is, as we stated
# initially, determined by the real world problem. Men and women suffer
# different medical conditions at different rates; young people and old
# people have different interests in advertised products; and different
# ethnicities experience different levels of systemic racism. These
# differences cannot be eliminated by a model.

# In general, the analysis above shows that it is impossible to achieve
# simultaneous equality in precision, true positive rate, and false
# positive rate. If we set our parameters so that our model is fair for
# two of these error functions, then we always find the condition in
# `[eq:practice:precisionequality] <#eq:practice:precisionequality>`__ as
# a consequence of the third. Unless all the positive and negative classes
# occur at the same rate for both classes, then achieving fairness in all
# three error functions is impossible. The result above has been refined
# by Kleinberg and colleagues, where they include properties of the
# classifier, :math:`f(x)`, in their derivation
# :raw-latex:`\parencite{kleinberg2018algorithmic}`.

# Various methods have been suggested by researchers to attempt to achieve
# results as close as possible to all three fairness criteria. We do not,
# however, discuss them here, for one simple reason. We wish to emphasise
# that solving ‘fairness’ is not primarily a technical problem. The ethics
# through awareness paradigm emphasises our responsibility as engineers to
# be aware of these limitations and explain them to clients, and a joint
# decision should be made on how to navigate the pitfalls.

# .. _`ch12:sec12.2`: