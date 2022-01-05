Probability Score Class Docs
============================
There is just one class to represent predictions returning probability scores

*  **BinaryScore**: an object that represents predictions matching each observation with a probability score between 0 and 1.

Probability scores can be easily transformed in predictions by setting a
threshold above which the observations are mapped to "1", while the remaining
get a "0".

BinaryScore
--------------------------------

.. automodule:: easypred.binary_score
   :members:
   :undoc-members:
   :show-inheritance: