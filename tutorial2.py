import os
import sys
import numpy
import theano
import theano.tensor as T

if __name__ == '__main__':
#declare shared variable
#http://deeplearning.net/software/theano/tutorial/examples.html#using-shared-variables
    a=theano.shared(0.01)
    b=T.dscalar("b")
    c=(a-b)*(a-b)+1
#set dc as the derivative of c with a
    dc=T.grad(c,a)
#declare function with updates.
#when function is called ,a is set to a-0.1*dc
    g=theano.function([b],c,updates=[[a,a-0.1*dc]])
    for i in xrange(50):
        print g(2),a.get_value()
