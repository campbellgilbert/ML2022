# Name: Campbell Gilbert
# COMP 347 - Machine Learning
# HW No. 4

#*********************
# PROBLEMS ARE POSTED BELOW THE FUNCTIONS SECTION!!!
#*********************

# Libraries
#------------------------------------------------------------------------------
from collections import Counter
from itertools import combinations
import numpy as np
import scipy.linalg as LA
import pandas            as pd
import matplotlib.pyplot as plt

# Functions
#------------------------------------------------------------------------------
def fact(n):
    #Factorial of an integer n>=0.
    if n in [0,1]:
        return 1
    else:
        return n*fact(n-1)

def partition(number:int, max_vals:tuple):
    S = set(combinations((k for i,val in enumerate(max_vals) for k in [i]*val), number))
    for s in S:
        c = Counter(s)
        yield tuple([c[n] for n in range(len(max_vals))])

def RBF_Approx(X,gamma,deg):
    #Transforms data in X to its RBF representation, but as an approximation
    #in deg degrees.  gamma = 1/2.
    new_X = []; N = X.shape[0]; n = X.shape[1]; count = 0
    for i in range(N):
        vec = []
        for k in range(deg+1):
            if k == 0:
                vec += [1]
            else:
                tup = (k,)*n
                parts = list(partition(k, tup))
                for part in parts:
                    vec += [np.prod([np.sqrt(gamma**deg)*(X[i,s]**part[s])/np.sqrt(fact(part[s])) for s in range(n)])]
        new_X += [np.exp(-gamma*LA.norm(X[i,:])**2)*np.asarray(vec)]
        print(str(count) + " of " + str(N))
        count += 1

    return np.asarray(new_X)

def smo_algorithm(X,y,C, max_iter, thresh):
    """Optimizes Lagrange multipliers in the dual formulation of SVM.
        X: The data set of size Nxn where N is the number of observations and
           n is the length of each feature vector.
        y: The class labels with values +/-1 corresponding to the feature vectors.
        C: A threshold positive value for the size of each lagrange multiplier.
           In other words 0<= a_i <= C for each i.
        max_iter: The maximum number of successive iterations to attempt when
                  updating the multipliers.  The multipliers are randomly selected
                  as pairs a_i and a_j at each iteration and updates these according
                  to a systematic procedure of thresholding and various checks.
                  A counter is incremented if an update is less than the value
                  thresh from its previous iteration.  max_iter is the maximum
                  value this counter attains before the algorithm terminates.
        thresh: The minimum threshold difference between an update to a multiplier
                and its previous iteration.
    """
    alph = np.zeros(len(y)); b = 0
    count = 0
    while count < max_iter:

        num_changes = 0

        for i in range(len(y)):
            #print("SMO i = ", i)
            w = np.dot(alph*y, X)
            E_i = np.dot(w, X[i,:]) + b - y[i]

            if (y[i]*E_i < -thresh and alph[i] < C) or (y[i]*E_i > thresh and alph[i] > 0):
                j = np.random.choice([m for m in range(len(y)) if m != i])
                E_j = np.dot(w, X[j,:]) + b - y[j]

                a_1old = alph[i]; a_2old = alph[j]
                y_1 = y[i]; y_2 = y[j]

                # Compute L and H
                if y_1 != y_2:
                    L = np.max([0, a_2old - a_1old])
                    H = np.min([C, C + a_2old - a_1old])
                elif y_1 == y_2:
                    L = np.max([0, a_1old + a_2old - C])
                    H = np.min([C, a_1old + a_2old])

                if L == H:
                    continue
                eta = 2*np.dot(X[i,:], X[j,:]) - LA.norm(X[i,:])**2 - LA.norm(X[j,:])**2
                if eta >= 0:
                    continue
                #Clip value of a_2
                a_2new = a_2old - y_2*(E_i - E_j)/eta
                if a_2new >= H:
                    a_2new = H
                elif a_2new < L:
                    a_2new = L

                if abs(a_2new - a_2old) < thresh:
                    continue

                a_1new = a_1old + y_1*y_2*(a_2old - a_2new)

                # Compute b
                b_1 = b - E_i - y_1*(a_1new - a_1old)*LA.norm(X[i,:]) - y_2*(a_2new - a_2old)*np.dot(X[i,:], X[j,:])
                b_2 = b - E_j - y_1*(a_1new - a_1old)*np.dot(X[i,:], X[j,:]) - y_2*(a_2new - a_2old)*LA.norm(X[j,:])

                if 0 < a_1new < C:
                    b = b_1
                elif 0 < a_2new < C:
                    b = b_2
                else:
                    b = (b_1 + b_2)/2

                num_changes += 1
                alph[i] = a_1new
                alph[j] = a_2new

        if num_changes == 0:
            count += 1
        else:
            count = 0
        print(count)
    return alph, b

def hinge_loss(X,y,w,b):
    """Here X is assumed to be Nxn where each row is a data point of length n and
    N is the number of data points.  y is the vector of class labels with values
    either +1 or -1.  w is the support vector and b the corresponding bias."""
    #for w the normal vector pointing perpendicularly out of the hyperplane
    # & b the scalar offset (if b = 0 then plane passes thru origin)
    # & r the distance of points away from the hyperplane:
    #if yi = +1 then <w, xi> + b >= r
    #if yi = -1 then <w, xi> + b <= -r

    yTilde = np.dot(y, np.dot(w, X) + b)
    return max(0, 1 - yTilde)

    #return FIXME

def hinge_deriv(X,y,w,b):
    print("calling hinge deriv")
    """Here X is assumed to be Nxn where each row is a data point of length n and
    N is the number of data points.  y is the vector of class labels with values
    either +1 or -1.  w is the support vector and b the corresponding bias."""
    N = len(X)
    gradientWSum = np.empty(len(w))
    gradientBSum = 0
    i = 0
    while (i < N):
        #if yi(wTxi+b) >= 1
        if (np.dot(y[i], np.dot(w, X[i])+b) >= 1):
            gradientWSum = np.add(gradientWSum, w)
        else:
            gradientWSum = np.add(gradientWSum, (w - (y[i] * X[i])))
            gradientBSum += -1 * y[i]
        i += 1

    gradW = (1/N) * gradientWSum
    gradB = (1/N) * gradientBSum

    return gradW, gradB


# Problem #1 - Hinge Loss Optimization for SVM on Randomized Test Data
#------------------------------------------------------------------------------
#In this problem you will be performing SVM using the hinge loss formalism as
#presented in lecture.

# 1a. Complete the function hinge_loss and hinge_deriv in the previous section.

# 1b. Perform SVM Using the hinge loss formalism on the data in svm_test_2.csv.
#     Use an appropriate initialization to your gradient descent algorithm.

def gradSVM(X, y, w, K, b):
    print("calling SVM")
    #x: inputs
    #y: outputs
    #w: normal vector to hyperplane
    #K: tolerance, signifying acceptable derivative norm for stopping descent
    #b: scalar offset (if b = 0, plane passes thru origin)

    step = 0.1
    i = 0
    gradW, gradB = hinge_deriv(X, y, w, b)
    while (i < 5000):
    #while (LA.norm(gradW) > K) or (gradB > K):
        gradW, gradB = hinge_deriv(X, y, w, b)
        w = np.subtract(w, step * gradW)
        b -= step * gradB
        i += 1

    return w, b

# 1c. Perform SVM on the data in svm_test_2.csv now using the Lagrange multiplier
#     formalism by calling the function smo_algorithm presented above.  Optimize
#     this for values of C = 0.25, 0.5, 0.75, and 1.  I recommend taking
#     max_iter = 2500 and thresh = 1e-5 when calling the smo_algorithm.

# 1d. Make a scatter plot of the data with decision boundary lines indicating
#     the hinge model, and the various Lagrange models found from part c.  Make
#     sure you have a legend displaying each one clearly and adjust the transparency
#     as needed.


inputSVM = pd.read_excel('svm_test_2.xlsx', usecols="A, B, C")
data = np.array(inputSVM)

X = data[:, :2]
y = data[:, 2]

plt.plot(data[0:200, 0], data[0:200, 1], '.',) #pos; below SVM line
plt.plot(data[201:, 0], data[201:, 1], 'x',) #neg; above SVM line

w, b = gradSVM(X, y, np.array([-10, 10]), 0.01, 10)
m = w.T[0]
plt.plot(data[:, 0], m*data[:, 0]+b, linewidth = 2.0, label='svm')

print("TESTING ONLY SMO")
cArray = [.25, .5, .75, 1]
max_iter = 2500
thresh = 1e-5
N = len(X)
"""
i = 0
while (i < 4):
    print("calling SMO...")
    alph, b = smo_algorithm(X, y, cArray[i], max_iter, thresh)
    j = 0
    wSum = np.empty((1, 2))
    while (j < N):

        wSum += alph[j] * y[j] * X[j]
        j += 1

    m = wSum.T[0]
    b = wSum.T[1]
    plt.plot(data[:, 0], m*data[:, 0]+b, linewidth = 2.0, label='C = {}'.format(cArray[i]))
    i += 1
"""
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('SVM test data')
plt.legend(loc = 'upper left')
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()



# 1e. Perform SVM on the radial data, but preprocess the data by using a kernel
#     embedding of the data into 3 dimensions.  This can be accomplished by
#     taking z = sqrt(x**2 + y**2) for each data point.  Learn an optimal model
#     for separating the data using the Lagrange multiplier formalism.  Experiment
#     with choices for C, max_iter, and thresh as desired.
"""
def transform(X):
    #Takes matrix of 2d data, turns it into matrix of 3d data
    zFull = np.zeros((len(X), 3))
    i = 0
    while (i < len(X)):
        x1 = X[i, 0]
        x2 = X[i, 1]
        zFull[i, 0] = x1
        zFull[i, 1] = x2
        zFull[i, 2] = np.sqrt(x1**2 + x2**2)
        i += 1

    return zFull

inputRad = pd.read_excel('radial_data.xlsx', usecols="A, B, C")
radialData = np.array(inputRad)

X = radialData[:, :2]
y = radialData[:, 2]

fig = plt.figure()
ax = plt.axes(projection='3d')

C=1
max_iter = 2500
thresh = 1e-5
zFull = transform(X)
alph, b = smo_algorithm(zFull, y, C, max_iter, thresh)

j = 0
wSumo = []
while (j < len(X)):
    if (alph[j] > 0.0):
        wSumo.append(alph[j] * y[j] * zFull[j])
    j += 1

ax.scatter(xs = zFull[:, 0], ys = zFull[:, 1], zs = zFull[:, 2], c = y)

wSum = np.asarray(wSumo)
X1, X2, X3 = np.meshgrid(wSum[0], wSum[1], wSum[2])
ax.plot_surface(X1.T[0], X2.T[0], X3.T[0])
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('z')
ax.set_title('Radial test data 3d')
plt.show()

"""
# Problem #2 - Cross Validation and Testing for Breast Cancer Data
#------------------------------------------------------------------------------
# In this problem you will use the breast cancer data in an attempt to use SVM
# for a real-world classification problem.

# 2a. Pre-process the data so that you separate the main variables.  Your data
#     X should consist all but the first two and very last columns in the dataframe.
#     Create a variable Y to reinterpret the binary classifiers 'B' and 'M' as
#     -1 and +1, respectively.

inputCancer = pd.read_excel('breast_cancer.xlsx')
cancerData = np.array(inputCancer)
allData = cancerData[:, 1:31]

classes = cancerData[:, 1]
i = 0
y = np.empty(len(classes))
while (i < len(classes)):
    if (classes[i] == 'M'):
        allData[i, 0] = 1
    else:
        allData[i, 0] = -1
    i += 1

# 2b. Perform cross-validation using a linear SVM model on the data by dividing
#     the indexes into 10 separate randomized classes (I recommend looking up
#     np.random.shuffle and np.array_split).  Make sure you do the following:
#       1. Make two empty lists, Trained_models and Success_rates.  In Trained_models
#          save ordered pairs of the learned models [w,b] for each set of training
#          training data.  In Success_rates, save the percentage of successfully
#          classified test points from the remaining partition of the data.  Remember
#          that the test for correct classification is that y(<w,x> + b) >= 1.
#       2. Make a histogram of your success rates.  Don't expect these to be stellar
#          numbers.  They will most likely be abysmal success rates.  Unfortunately
#          SVM is a difficult task to optimize by hand, which is why we are fortunate
#          to have kernel methods at our disposal.  Speaking of which.....
# 2c. Repeat cross-validation on the breast cancer data, but instead of a linear
#     SVM model, employ an approximation to the RBF kernel as discussed in class.
#     Note that what this does is that it transforms the original data x into a
#     variable X where the data is embedded in a higher dimension.  Generally when
#     data gets embedded in higher dimensions, there's more room for it to be spaced
#     out in and therefore increases the chances that your data will be linearly
#     separable.  Do this for deg = 2,3.  I recommend taking gamma = 1e-6.
#     Don't be surprised if this all takes well over an hour to terminate.


#1: shuffle data
#2: apply pre-processing: case 1 is linear, case 2 is rbf deg 2, case 3 is rbf deg 3
"""
#change to test RBF for breast cancer data
METHOD = 'LIN'
DEG = 2

np.random.shuffle(allData)
totalSets = np.array_split(allData, 10)
trainingSets = totalSets[0:9]
testSet = totalSets[9]

i = 0
#trainedModels = np.zeros((9, 30))
max_iter = 500
thresh = 1e-300
wStart = np.random.rand(29) * 10


indexList = np.arange()

while (i < 9):
    x = trainingSets[i][:, 1:]
    y = trainingSets[i ][:, 0]

    if (METHOD == 'LIN'):
        print("using method lin")
        X = x
        w, b = gradSVM(X, y, wStart, 0.01, 10)


    elif (METHOD == 'RBF'):
        w = np.random.rand(465) * 10
        #cases = [2, 3]
        k = 0
        while (k < 2):
            #print("deg = ", cases[k])
            #DEG = cases[k]
            gamma = 1e-6
            X = RBF_Approx(x,gamma,DEG)
            print("calling SMO...")
            alph, b = smo_algorithm(X, y, .75, 500, thresh)
            print("done w SMO")
            w = np.empty((1, 465))
            j = 0
            while (j < len(X)):
                hold = float(np.dot(alph[j], y[j]))
                adder = alph[j] * y[j] * X[j]
                #print("adder shape: ", np.shape(adder))
                #print("w shape: ", np.shape(w))
                w += adder
                j += 1
            k += 1
    trainedModels[i, 0] = b
    trainedModels[i, 1:30] = w
    i += 1

X = testSet[:, 1:]
y = testSet[:, 0]
successRates = np.zeros(9)
i = 0
while (i < 9):
    w = trainedModels[i, 1:30]
    b = trainedModels[i, 0]
    if (np.dot(y, (np.dot(w, X.T) + b).T) >= 1):
        #save percentage of successfully classified test points
        successRates[i] = np.dot(y, (np.dot(w, X.T) + b).T) - 1
    else:
        successRates[i] = 0
    i += 1

plt.hist(successRates)
plt.xlabel('Success Rate')
plt.ylabel('Frequency')
plt.title('Breast cancer SVM histogram')
plt.show()
"""
# Notes for Problem #2:
# 1. To save yourself from writing the same code twice, I recommend making this
#    type of if/else statement before performing SVM on the breast cancer data:

#        METHOD = ''
#        if METHOD == 'Lin':
#            X = x
#        elif METHOD == 'RBF':
#            deg = 2; gamma = 1e-6

#            X = RBF_Approx(x,gamma,deg)

# 2. For implementing smo_algorithm for the breast cancer data, I recommend
#    taking max_iter = 500 and thresh = 1e-300
