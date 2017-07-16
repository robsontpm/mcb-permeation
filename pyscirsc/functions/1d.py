"""
One dimensional piecewise linear functions
"""

__author__ = 'robson'


# Obsolete: implementation dumped for some reason...

class function:
    """
    One dimensional piecewise linear function
    """
    def __init__(self, x, y):
        """
        constructor
        :param x iterable arguments of the function (domain)
        :param y iterable value of the function in corresponding x y[i] = f(x[i])
        """
        self.x = x
        self.y = y


