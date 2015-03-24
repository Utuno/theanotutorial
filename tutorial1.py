# coding: UTF-8
import os
import sys
import numpy
import theano
import theano.tensor as T

if __name__ == '__main__':
    #declare variable
    #http://deeplearning.net/software/theano/library/tensor/basic.html
    x=T.dvector("x") 
    y=T.iscalar("y") 
    #declare function
    #http://deeplearning.net/software/theano/library/compile/function.html
    f=theano.function([x,y],x*y)
    
    #execute function
    print f([1,2],-1)

