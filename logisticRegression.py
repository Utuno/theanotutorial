#http://deeplearning.net/software/theano/tutorial/examples.html#a-real-example-logistic-regression
import numpy
import theano
import theano.tensor as T
rng = numpy.random

N = 10
feats = 10
realW=numpy.array([1]*10)
realB=-1
X=rng.randn(N, feats)
Y=[1 if realW.dot(x)+realB>0 else 0 for x in X]

D=[X,Y]
print D[0][:min(10,N)],D[1][:min(10,N)]

training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
print "Initial model:"
print w.get_value(), b.get_value()

# Construct Theano expression graph
p_1 = T.nnet.sigmoid(-T.dot(x, w) - b)   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print "Final model:"
print w.get_value(), b.get_value()
print "target values for D:", D[1]
print "prediction on D:", predict(D[0])

