# Name: Campbell Gilbert!Q
# COMP 347 - Machine Learning
# HW No. 3

# Libraries
#------------------------------------------------------------------------------
import numpy             as np
import scipy.linalg      as LA
import time
import pandas            as pd
import matplotlib.pyplot as plt
import math
import random

# Problem 1 - Gradient Descent Using Athens Temperature Data
#------------------------------------------------------------------------------
# For this problem you will be implementing various forms of gradient descent
# using the Athens temperature data.  Feel free to copy over any functions you
# wrote in HW #2 for this.  WARNING: In order to get gradient descent to work
# well, I highly recommend rewriting your cost function so that you are dividing
# by N (i.e. the number of data points there are).  Carry this over into all
# corresponding expression where this would appear (the derivative being one of them).

def A_mat(x, deg):
    """Create the matrix A part of the least squares problem.
    x: vector of input data.
    deg: degree of the polynomial fit.
    """
    A = np.ones((len(x), deg + 1))
    i = 0
    while i < len(x):
        j = 0
        while j < (deg + 1):
            A[i, j] = x[i]**(deg - j)
            j += 1
        i += 1
    return A

def LLS_func(x,y,w,deg):
    #FIXME -- possible error w/ dividing by N
    """The linear least squares objective function.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights.
       deg: degree of the polynomial."""
    #||Aw - y||^2
    A = A_mat(x, deg)
    AwMinusY = np.dot(A, w) - y
    f = np.dot(AwMinusY, AwMinusY)
    return f / x.size

def LLS_Solve(x,y, deg):
    """Find the vector w that solves the least squares regression.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit."""

    A = A_mat(x, deg)
    ATAinv = np.linalg.inv(np.dot(A.T, A))
    w = np.dot(ATAinv, np.dot(A.T, y))
    return w

# 1a. Fill out the function for the derivative of the least-squares cost function:

def LLS_deriv(x,y,w,deg):
    """Computes the derivative of the least squares cost function with input
    data x, output data y, coefficient vector w, and deg (the degree of the
    least squares model)."""
    # f′(w) = 2A^T (Aw − y)
    A = A_mat(x, deg)
    TwoAT = (2/x.size) * A.T
    AwMinusY = np.subtract(np.dot(A, w), y)
    f = np.dot(TwoAT, AwMinusY)
    return f

# 1b. Implement gradient descent as a means of optimizing the least squares cost
#     function.  Your method should include the following:
#       a. initial vector w that you are optimizing,
#       b. a tolerance K signifying the acceptable derivative norm for stopping
#           the descent method,
#       c. initial derivative vector D (initialization at least for the sake of
#           starting the loop),
#       d. an empty list called d_hist which captures the size (i.e. norm) of your
#           derivative vector at each iteration,
#       e. an empty list called c_hist which captures the cost (i.e. value of
#           the cost function) at each iteration,
#       f. implement backtracking line search as part of your steepest descent
#           algorithm.  You can implement this on your own if you're feeling
#           cavalier, or if you'd like here's a snippet of what I used in mine:
#
"""              eps = 1
#                m = LA.norm(D)**2
#                t = 0.5*m
#                while LLS_func(a_min, a_max, w - eps*D, 1) > LLS_func(a_min, a_max, w, 1) - eps*t:
#                    eps *= 0.9
"""
#       Plot curves showing the derivative size (i.e. d_hist) and cost (i.e. c_hist)
#       with respect to the number of iterations.

def stepSize(x, y, w, D, eps):
    m = np.dot(D, D)
    t = 0.5*m

    while LLS_func(x, y, w - (eps*D), 1) > (LLS_func(x, y, w, 1) - eps*t):
        eps *= 0.9
    return eps

def gradientStandard(w, K, x, y, deg, D):
    """#w: init vector being optimized
    #K: tolerance, signifying acceptable derivative norm for stopping descent
    #D: init deriv vector (to start loop)
    #y: outputs
    #x: inputs
    #d_hist: empty list that captures size (norm) of derivative vector at each iteration
    #c_hist: empty list that captures cost (value of cost function) at each iteration"""
    d_hist = []
    c_hist = []

    fPrime = D
    wCurr = w
    eps = 1
    i = 0
    while (LA.norm(fPrime) >= K):
        fPrime = LLS_deriv(x, y, wCurr, deg)

        epsNew = stepSize(x, y, wCurr, fPrime, eps)
        eps = epsNew

        wNext = wCurr - eps*fPrime
        wCurr = wNext

        d_hist.append(LA.norm(fPrime))
        c_hist.append(LLS_func(x, y, wCurr, deg))
        i += 1

    return wCurr, c_hist, d_hist


inputAthensMaxMin = pd.read_excel('athens-data.xlsx', usecols="H, I")
dataAthens = np.array(inputAthensMaxMin).T

K = 0.01
wStart = np.array([100, -100])
dStart = LLS_deriv(dataAthens[1], dataAthens[0], wStart, 1)

w, c_hist, d_hist = gradientStandard(wStart, K, dataAthens[1], dataAthens[0], 1, dStart)
m = w[0]
b = w[1]
#print(w)

plt.plot(dataAthens[1], dataAthens[0], '.', label='Temps')
plt.plot(dataAthens[1], m*dataAthens[1]+b, linewidth = 2.0, label='Gradient Descent')
plt.xlabel('Min Temp (C)')
plt.ylabel('Max Temp (C)')
plt.title('Temperature in Athens 1944-1945 w/ Gradient Descent')
plt.legend(loc = 'upper left')
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()

plt.plot(c_hist)
plt.xlabel('Iterations')
plt.ylabel('Cost of graddesc per iteration')
plt.title('c_hist for gradient descent')
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()

plt.plot(d_hist)
plt.xlabel('Iterations')
plt.ylabel('Norm of derivative at each iteration')
plt.title('d_hist for gradient descent')
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()


# 1c. Repeat part 1b, but now implement mini-batch gradient descent by randomizing
#     the data points used in each iteration.  Perform mini-batch descent for batches
#     of size 5, 10, 25, and 50 data points.  For each one, plot the curves
#     for d_hist and c_hist.  Plot these all on one graph showing the relationship
#     between batch size and convergence speed (at least for the least squares
#     problem).  Feel free to adjust the transparency of your curves so that
#     they are easily distinguishable.

x = dataAthens[1]
y = dataAthens[0]
sizes = [5, 10, 25, 50]
fig, axs = plt.subplots(3, 2, constrained_layout=True)

axs[2, 0].set_title('c_hists for diff gradient descents')
axs[2, 1].set_title('d_hist for diff gradient descents')

i = 0
row = 0
while (row < 2):
    col = 0
    while (col < 2):
        batchNums = np.random.choice(x.size, sizes[i])
        j = 0
        xBatch = np.empty(sizes[i])
        yBatch = np.empty(sizes[i])
        while (j < sizes[i]):
            xBatch[j] = x[batchNums[j]]
            yBatch[j] = y[batchNums[j]]
            j += 1

        w, c_hist, d_hist = gradientStandard(wStart, K, xBatch, yBatch, 1, dStart)
        m = w[0]
        b = w[1]
        #print(w)
        axs[row, col].set_title('Temperature in Athens 1944-1945 w/ Gradient Descent for Batch Size {}'.format(sizes[i]))
        axs[row, col].plot(xBatch, yBatch, '.', label='Temps')
        axs[row, col].plot(xBatch, m*xBatch+b, linewidth = 2.0, label='Gradient Descent')
        axs[row, col].set_xlabel('Min Temp (C)')
        axs[row, col].set_ylabel('Max Temp (C)')
        axs[row, col].legend(loc = 'upper left')

        axs[2, 0].plot(c_hist, label='{}'.format(sizes[i]))
        axs[2, 1].plot(d_hist, label='{}'.format(sizes[i]))

        i += 1
        col += 1
    row += 1

axs[2, 0].set_xlabel('Iterations')
axs[2, 0].set_ylabel('Cost of graddesc per iteration')
axs[2, 0].legend(loc = 'upper left')

axs[2, 1].set_xlabel('Iterations')
axs[2, 1].set_ylabel('Norm of derivative at each iteration')
axs[2, 1].legend(loc = 'upper left')

plt.show()


# 1d. Repeat 1b, but now implement stochastic gradient descent.  Plot the curves
#     for d_hist and c_hist.  WARNING: There is a strong possibility that your
#     cost and derivative definitions may not compute the values correctly for
#     for the 1-dimensional case.  If needed, make sure that you adjust these functions
#     to accommodate a single data point.

def LLS_func_stoch(x,y,w):
    """The linear least squares objective function.
       x: vector of input data. In this case, only 1 datapoint.
       y: vector of output data. In this case, only 1 datapoint.
       w: vector of weights.
       deg: degree of the polynomial."""
    xFloat = float(x)
    A = np.matrix([xFloat, 1])
    AwMinusY = np.dot(A, w) - y
    AwMinusYRet = AwMinusY.getA1()
    f = np.dot(AwMinusYRet, AwMinusYRet)
    return f

def LLS_deriv_stoch(x,y,w):
    """Computes the derivative of the least squares cost function with input
    data x, output data y, coefficient vector w, and deg (the degree of the
    least squares model)."""
    # f′(w) = 2A^T (Aw − y)
    xFloat = float(x)
    A = np.matrix([xFloat, 1])
    TwoAT = 2 * A.T
    AwMinusY = np.subtract(np.matmul(A, w), y)
    f = np.dot(TwoAT, AwMinusY)
    fRet = f.getA1()
    return fRet

def stepSizeStoch(x, y, w, D, eps, i):
    m = np.dot(D, D)
    t = 0.5*m

    while LLS_func_stoch(x, y, w - (eps*D)) > (LLS_func_stoch(x, y, w) - eps*t):
        eps *= 0.9
    return eps

def gradientStoch(w, K, x, y, D):
    """#w: init vector being optimized
    #K: tolerance, signifying acceptable derivative norm for stopping descent
    #D: init deriv vector (to start loop)
    #y: outputs
    #x: inputs
    #d_hist: empty list that captures size (norm) of derivative vector at each iteration
    #c_hist: empty list that captures cost (value of cost function) at each iteration"""
    d_hist = []
    c_hist = []

    fPrime = D
    wCurr = w
    eps = 1
    i = 0
    while (LA.norm(fPrime) >= K):
        choiceNum = np.random.choice(x.size, 1)
        xChoice = x[choiceNum]
        yChoice = y[choiceNum]
        fPrime = LLS_deriv_stoch(xChoice, yChoice, wCurr)

        epsNew = stepSizeStoch(xChoice, yChoice, wCurr, fPrime, eps, i)
        eps = epsNew

        wNext = wCurr - eps*fPrime
        wCurr = wNext

        d_hist.append(LA.norm(fPrime))
        c_hist.append(LLS_func_stoch(xChoice, yChoice, wCurr))
        i += 1

    return wCurr, c_hist, d_hist

K = 0.01
wStart = np.array([100, -100])
dStart = [-1, 1]

w, c_hist, d_hist = gradientStoch(wStart, K, dataAthens[1], dataAthens[0], dStart)
m = w[0]
b = w[1]

plt.plot(dataAthens[1], dataAthens[0], '.', label='Temps')
plt.plot(dataAthens[1], m*dataAthens[1]+b, linewidth = 2.0, label='Stochastic gradient Descent')
plt.xlabel('Min Temp (C)')
plt.ylabel('Max Temp (C)')
plt.title('Temperature in Athens 1944-1945 w/ Stochastic Gradient Descent')
plt.legend(loc = 'upper left')
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()

plt.plot(c_hist)
plt.xlabel('Iterations')
plt.ylabel('Cost of graddesc per iteration')
plt.title('c_hist for stochastic gradient descent')
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()

plt.plot(d_hist)
plt.xlabel('Iterations')
plt.ylabel('Norm of derivative at each iteration')
plt.title('d_hist for stochastic gradient descent')
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()


# 1e. Aggregate your curves for batch, mini-batch, and stochastic descent methods
#     into one final graph so that a full comparison between all methods can be
#     observed.  Make sure your legend clearly indicates the results of each
#     method.  Adjust the transparency of the curves as needed.

x = dataAthens[1]
y = dataAthens[0]
K = 0.01
wStart = np.array([100, -100])
dStart = [-1, 1]

sizes = [x.size, 5, 10, 25, 50, 1]
sizeLabels = ["full", "5", "10", "25", "50", "stochastic"]

#making 2 plots at once
fig, axs = plt.subplots(3, 1, constrained_layout=True)

#print("now working on batch of size ", sizes[i])
axs[0].set_title('Temperature in Athens 1944-1945 w/ Gradient Descents')
axs[1].set_title('c_hists for diff gradient descents')
axs[2].set_title('d_hist for diff gradient descents')
i = 0
while (i < 6):
    batchNums = np.random.choice(x.size, sizes[i])
    xBatch = np.empty(sizes[i])
    yBatch = np.empty(sizes[i])
    j = 0
    while (j < sizes[i]):
        xBatch[j] = x[batchNums[j]]
        yBatch[j] = y[batchNums[j]]
        j += 1

    if (i < 5):
        w, c_hist, d_hist = gradientStandard(wStart, K, xBatch, yBatch, 1, dStart)
    else:
        w, c_hist, d_hist = gradientStoch(wStart, K, xBatch, yBatch, dStart)

    m = w[0]
    b = w[1]

    axs[0].plot(xBatch, m*xBatch+b, linewidth = 2.0, label='Gradient Descent for size {}'.format(sizeLabels[i]))
    axs[1].plot(c_hist, label='{}'.format(sizeLabels[i]))
    axs[2].plot(d_hist, label='{}'.format(sizeLabels[i]))

    i += 1

axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Cost of graddesc per iteration')
axs[1].legend(loc = 'upper left')

axs[2].set_xlabel('Iterations')
axs[2].set_ylabel('Norm of derivative at each iteration')
axs[2].legend(loc = 'upper left')

axs[0].plot(x, y, '.', label='Temps')
axs[0].set_xlabel('Min Temp (C)')
axs[0].set_ylabel('Max Temp (C)')
axs[0].legend(loc = 'upper left')


plt.show()


# Problem 2 - LASSO Regularization
#------------------------------------------------------------------------------
# For this problem you will be implementing LASSO regression on the Yosemite data.

# 2a. Fill out the function for the soft-thresholding operator S_lambda as discussed
#     in lecture:

def soft_thresh(v, lam):
    #Perform the soft-thresholding operation of the vector v using parameter lam.
    #operations are performed entrywise on input vector?
    retV = []
    for vi in v:
        if (vi > lam):
            retV.append(vi - lam)
        elif (np.abs(vi) <= lam):
            retV.append(0)
        else:
            retV.append(vi + lam)
    return retV


# 2b. Using 5 years of the Yosemite data, perform LASSO regression with the values
#     of lam ranging from 0.25 up to 5, spacing them in increments of 0.25.
#     Specifically do this for a cubic model of the Yosemite data.  In doing this
#     save each of your optimal parameter vectors w to a list as well as solving
#     for the exact solution for the least squares problem.  Make the following
#     graphs:
#
#       a. Make a graph of the l^2 norms (i.e. Euclidean) and l^1 norms of the
#          optimal parameter vectors w as a function of the coefficient lam.
#          Interpret lam = 0 as the exact solution.  One can find the 1-norm of
#          a vector using LA.norm(w, ord = 1)
#       b. For each coefficient in the cubic model (i.e. there are 4 of these),
#           make a separate plot of the absolute value of the coefficient as a
#           function of the parameter lam (again, lam = 0 should be the exact
#           solution to the original least squares problem).  Is there a
#           discernible trend of the sizes of our entries for increasing values
#           of lam?

def fLassoCost(x,y,w,deg,norm):
    """The linear least squares objective function.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights.
       deg: degree of the polynomial."""
    #||Aw - y||^2 + l1 norm
    lNorm = LA.norm(w, norm)
    return LLS_func(x, y, w, deg) + lNorm

def fLasso(x, y, w, K, deg, D, lam):
    fPrime = D
    wCurr = w
    i = 0
    A = A_mat(x, deg)
    wPrev = []
    while (LA.norm(fPrime, 2) >= K and i <= 8000):
        v = np.subtract(wCurr, (lam * LLS_deriv(x, y, w, deg)))
        wNext = soft_thresh(v, lam)

        wCurr = wNext
        fPrime = LLS_deriv(x, y, wCurr, deg) + LA.norm(wCurr, 1)

        i += 1

    return wCurr

x = dataAthens[1]
y = dataAthens[0]
K = 0.01
wStart = np.array([100, -100])
dStart = LLS_deriv(x, y, wStart, 1)

lams = np.arange(0.1/x.size, 1.5/x.size, 0.1/x.size)
print("lams: ", lams)
norms = np.array([1, 2])

opts = np.empty((14, 2))
norms1 = np.empty(14)
norms2 = np.empty(14)

opts[0] = LLS_Solve(x, y, 1)
norms1[0] = LA.norm(opts[0], ord=1)
norms2[0] = LA.norm(opts[0])

i = 0
while (i < 14):
    w = fLasso(x, y, wStart, K, 1, dStart, lams[i])
    m = w[0]
    b = w[1]
    opts[i] = w
    norms1[i] = LA.norm(opts[i], ord = 1)
    norms2[i] = LA.norm(opts[i])
    i += 1

plt.xlabel('Lambda')
plt.ylabel('Norm size')
plt.title('norm of w vs lam')
plt.plot(lams, norms1, label='l1')
plt.plot(lams, norms2, label='l2')
plt.legend(loc = 'upper left')
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()

plt.xlabel('Lambda')
plt.ylabel('abs val of coeffs')
plt.title('coeffsize vs .lam')
plt.plot(lams, np.abs(opts[:, 0]), label = ("Linear Coeffs"))
plt.plot(lams, np.abs(opts[:, 1]), label = ("constant Coeffs"))
plt.legend(loc = 'upper left')
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()
