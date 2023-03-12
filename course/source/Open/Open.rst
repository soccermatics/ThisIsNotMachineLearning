.. role:: raw-latex(raw)
   :format: latex
..

Being open to different views of a complex system.

There is no definitive methodology for modelling of complex systems,
just a set of plural practices.

In this chapter we focus on several application areas.

Animal behaviour
================

The multiple views approach is also adopted when, for example, ant
pheromone trails are modelled in terms of cycles of ant activity,
formation and topology of the spatial patterns of trail networks,
evolution of co-operation and chemical properties of the
trails :raw-latex:`\cite{sumpter2010collective}`. Further examples are
found in modelling the growth of tumours, genetic networks and
ecological systems. Multiple views are also a prerequisite for modelling
(more complex) human social
systems :raw-latex:`\cite{helbing2010pluralistic}`. In adopting an Open
ML approach, we simultaneously engage many different frameworks and
views of a system, each designed to answer a different sub-question. We
take different snapshots of the system and then use each of them to
construct a bigger picture of the system. The more snapshots we include,
the more complete the bigger picture. ML might help find the sharpest
focus of one particular snapshot, but it can not tell us what is a good,
overall picture.

Football
========

Team sports are more complex compared to board games, for example. They
involve social, physical, tactical, and mental aspects. Team sports are
however less complex than other systems such as human societies,
financial systems, or human brains. Modelling the game of football, thus
allows us to understand some of the challenges involved in modelling
open systems, while still dealing with an application of (somewhat)
limited scope.

.. figure:: Source/images/Illustration_old.png
   :alt: 
   :width: 11.5cm

A widely used model for predicting the outcome of a football match is
Poisson regression :raw-latex:`\cite{dixon1997modelling}`. The central
idea is that goals in the match are independent, occurring at a rate
which depends on the relative quality of the teams and which can be
estimated using regression methods. This model is used by professional
gamblers and bookmakers, since it outperforms betting strategies of the
customers of the bookmakers (see
e.g. :raw-latex:`\cite{spann2009sports}`). It is possible to include
more factors, including events during the match, for example, in a
neural network to improve predictions, giving a *prediction* view of the
game.

The prediction view is of little use to the players, who will have some
sense of the strength of their opponents, and thus whether or not their
team is likely to win, but can’t be helped by a model (ML or otherwise)
which sets probabilities to the outcome. Those playing the game want to
understand specific details of their opponents’ and their teammates’
play which they can exploit during the match. Models that provide these
insights can be found, with help of ML, through concepts such as pass
probability and pass values, which (using historical data) evaluate the
quality of
actions :raw-latex:`\cite{fernandez2019decomposing,sumpter2016soccermatics}`.

There are many other levels and dimensions to football, as
Figure `1 <#football>`__ shows. For example, the *bio-mechanics* view
looks at the body kinematics of
players :raw-latex:`\cite{ibrahim2019kinematic}`. One example of the
*societal* view is statistical analysis of refereeing to reveal
discrimination in decisions made :raw-latex:`\cite{gallo2013punishing}`.
Another is the use of computer vision to investigate how sports
commentators use words, such as ‘pace’ and ‘power’, when describing
players with non-white backgrounds while words such as ‘hard work’,
’effort’ and ’mental skill’ are used to describe white players. For
example, :raw-latex:`\cite{Gregory2021Pace}` looked at how commentators
described events on the pitch when they could and couldn’t identify the
ethnicity of the players.

Closed and Partially Open ML models can be, and are used in the approach
outlined above, in the sense that regression, neural networks, and other
methods are used to fit data. But their usage is secondary to finding
different views of the sport, taken from different perspectives. Finding
a view is sometimes referred to in ML as feature selection. But this
terminology places the ML model as primary and the features as
secondary. The problem with framing this process as feature selection is
that it gives the model itself an aura of neutrality to which
subjectively chosen features are added. In fact, the open-ended process
of model building is always a necessarily value-laden endeavour. The
Open ML approach, which we emphasize, places the ML model as a tool for
fitting data, once we have found the view we are interested in. Open ML,
then, is about finding a useful view for a certain problem, and
combining the views to get an overall understanding of the system. The
usefulness of the view subsequently cannot be entirely divorced from the
modeller’s objectives, motivations, and perspectives.

Human social behaviour
======================
