# Name: Campbell Gilbert
# COMP 347 - Machine Learning
# Final Project: Genetic Algorithm


# Libraries
#------------------------------------------------------------------------------
import numpy as np
import scipy.linalg as LA
import pandas            as pd
import matplotlib.pyplot as plt

import random


# Functions
#------------------------------------------------------------------------------

#NEW FUNCTIONS
def createPopulation(PopulationSize):
    """Initializes random population of organisms."""
    solutionSpace = [{
        "m": (random.random() - 0.5) * 10,
        "b": (random.random() - 0.5) * 10,
    } for i in range(PopulationSize)]
    return solutionSpace

def calcError(org, case, X, Y):
    """Gets fitness of given organism based on error. Lower/more negative error
    means the organism is better-fit for the problem.
    The most important part of the GA."""
    if (case == "linear"):
        err = 0
        for x, y, in zip(X, Y):
            #error is distance between point and point on line
            #we want this distance to be as small as possible
            y_ = org["m"] * x + org["b"]
            err = err + (y - y_)**2
    elif (case == "classify"):
        err = 0
        for x, y in zip(X, Y):
            if (y <= 0): #negative, should be above line
            #we want x2 > x2_
            #error is x2_ - x2 (we want this to be small or negative)
                x1 = x[0]
                x2 = x[1] #"y"-value
                x2_ = org["m"] * x1 + org["b"] #"y"-value if point was ON the line
                err = err + (x2_-x2)
            else: #positive, should be below line
            #we want x2_ > x2, error should be x2 - x2_ (we want this to be small or negative)
                x1 = x[0]
                x2 = x[1] #"y"-value

                x2_ = org["m"] * x1 + org["b"] #"y"-value if point was ON the line
                err = err + (x2-x2_)
    return err

def calcFitness(population, case, X, Y):
    """Calculate fitness of given population."""
    for org in population:
        org["fitness"] = calcError(org, case, X, Y)

    #Sort population
    population = sorted(population, key = lambda x:x["fitness"])
    #Print best fitting organism.
    print(
        "m:", population[0]["m"],
        "b:", population[0]["b"],
        "f:", population[0]["fitness"]
    )
    return population

def mate(mom, dad):
    """Create random chromosome given 2 possible parents and chance of mutation."""
    #init random chromosome
    kidChromosome = {
        "m": (random.random() - 0.5) * 10,
        "b": (random.random() - 0.5) * 10,
    }
    #40% chance of mom, 40% chance of dad, 20% chance of rand for either param
    for i in range(2):
        prob1 = random.random()
        if prob1 < 0.4:
        	kidChromosome["m"] = mom["m"]
        elif prob1 < 0.8:
        	kidChromosome["m"] = dad["m"]

        prob2 = random.random()
        if prob2 < 0.4:
        	kidChromosome["b"] = mom["b"]
        elif prob2 < 0.8:
        	kidChromosome["b"] = dad["b"]
    return kidChromosome


def runGA(X, Y, popSize, case, idealFitVal):
    """Runs GA simulation for given number of populations."""
    population = createPopulation(popSize)

    found = False
    iter = 0

    while not found:
        #sort population and calculate fitness
        population = calcFitness(population, case, X, Y)
        #print best candidate
        print("best candidate: ",
            "m:", population[0]["m"],
            "b:", population[0]["b"],
            "f:", population[0]["fitness"]
        )

        #check for lowest possible fitness
        if (population[0]["fitness"] <= idealFitVal):
            found = True
        break

        #create new generation
        newGen = []

        #top 10% of population succeeds to next gen
        size = int((10*popSize)/100)
        newGen.extend(population[:size])

        #mate better half of population
        for i in range(popSize):
            mom = random.choice(population[:50])
            dad = random.choice(population[:50])
            child = mate(mom, dad)
            newGen.append(child)

        population = newGen
        iter += 1

    population = calcFitness(population, case, X, Y)
    best = population[0]

    return best

#PREMADE (From past projects)
def dist_matrix(Data):
    """Returns the Euclidean distance matrix dists."""
    print("calling dist matrix")
    N = len(Data)
    print("N: ", N)
    distMat = np.zeros([N, N])
    i = 0
    while (i < N):
    #print("i: ", i)
        j = 0
        while (j < N):
            #print("j: ", j)

            if (i != j):
                distMat[i, j] = LA.norm(Data[i] - Data[j])
            j += 1
        i += 1
    return distMat

def classical_mds(Dists, dim):
    """Takes the distance matrix Dists and dimensionality dim of the desired
    output and returns the classical MDS compression of the data."""
    N = np.shape(Dists)[0]
    ones = np.ones((N,N))
    H = np.identity(N) - (1/N)*ones
    XXT = (-1/2)*np.dot(H, np.dot(Dists * Dists, H))
    P, D, PT = LA.svd(XXT)
    Y = np.dot(P, np.diag(np.sqrt(D)))
    return Y[:,:dim]

def edge_matrix(Dists,eps):
    """Returns the epsilon-ball edge matrix for a data set.  In particular, the
    edge matrix should be NxN with binary entries: 0 if the two points are not
    neighbors, and 1 if they are neighbors."""
    #inf if not neighbors, Dists[r, c] if neighbors
    N = len(Dists)
    edge = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            if Dists[i, j] <= eps:
                edge[i, j] = 1
                edge[j, i] = 1
    return edge

def isomap(dists, edges, dim):
    """Returns the Isomap compression of a data set based on the Euclidean distance
    matrix dists, the edge matrix edges, and the desired dimensionality dim of
    the output.  This should specifically output two variables: it should output
    the data compression, as well as the indices of points removed from the
    Floyd-Warshall algorithm."""
    N = len(edges)
    #fw = np.copy(dists)
    fw = dists
    inf = 99999
    pointsRemoved = []
    #create matrix
    for r in range(N):
        for c in range(r):
            if edges[r, c] == 0:
                fw[r, c] = inf
                fw[c, r] = inf
    #FW
    for k in range(N):
        print("k: ", k)
        for r in range(N):
            for c in range(r):
                fw[r, c] = min(dists[r, c], dists[r, k] + dists[k, c])
                fw[c, r] = fw[r, c]

    #remove disconnected points
    i = 0
    while (i < N):
        if fw[i, 0] == inf or fw[i, 1] == inf:
            pointsRemoved.append(i)
            fw = np.delete(fw, i, axis=0)
            fw = np.delete(fw, i, axis=1)
            N -= 1
            continue
        i += 1

    return classical_mds(fw, dim), np.array(pointsRemoved)

def hinge_loss(X, y, w, b):
    """Here X is assumed to be Nxn where each row is a data point of length n and
    N is the number of data points.  y is the vector of class labels with values
    either +1 or -1.  w is the support vector and b the corresponding bias."""
    """
    yTilde = np.dot(y, np.dot(w, X.T) + b)
    return max(0, 1 - yTilde)
    """
    N = len(X)
    i = 0
    worm = np.dot(w, w) / 2
    sum = 0
    while i < N:
        yTilde = np.dot(y[i], np.dot(w, X[i]) + b)
        sum += max(0, 1 - yTilde)
        i += 1
    return worm + sum/N

def hinge_deriv(X,y,w,b):
    """Here X is assumed to be Nxn where each row is a data point of length n and
    N is the number of data points.  y is the vector of class labels with values
    either +1 or -1.  w is the support vector and b the corresponding bias."""
    N = len(X)
    w = np.asarray(w)
    gradientWSum = np.zeros(len(w))
    gradientBSum = 0
    i = 0
    while (i < N):
        if (np.dot(y[i], np.dot(w, X[i])+b) >= 1):
            gradientWSum = np.add(gradientWSum, w)
        else:
            gradientWSum = gradientWSum + (w - (y[i] * X[i]))
            gradientBSum += -1 * y[i]
        i += 1
    gradW = (1/N) * gradientWSum
    gradB = (1/N) * gradientBSum
    return gradW, gradB

def stepSize(X, y, w, b, eps, dw, db):
    #b: scalar offset (if b = 0, plane passes thru origin)
    c = hinge_loss(X, y, w, b)
    cW = hinge_loss(X, y, w-eps*dw, b)
    cB = hinge_loss(X, y, w, b-eps*db)

    i = 0
    while (cW > (c - (eps*.5*np.dot(dw, dw)))) and (cB > (c - (eps*.5*np.dot(db, db)))) and i < 200:
        eps *= 0.9
        cW = hinge_loss(X, y, w - eps*dw, b)
        cB = hinge_loss(X, y, w, b-eps*db)
        i += 1
    return eps

def gradSVM(X, y, w, b):
    #x: inputs
    #y: outputs
    #w: normal vector to hyperplane
    #K: tolerance, signifying acceptable derivative norm for stopping descent
    #b: scalar offset (if b = 0, plane passes thru origin)

    gradW, gradB = hinge_deriv(X, y, w, b)
    step = stepSize(X, y, w, b, 0.1, gradW, gradB)

    gradWNext, gradBNext = hinge_deriv(X, y, w-step*gradW, b-step*gradB)
    #gradW, db1 = hinge_deriv(X, y, w, b-step*gradB)

    while (0.99 < (gradB/gradBNext) < 1.01) and not (0.99 < (LA.norm(gradW)/LA.norm(gradWNext)) < 1.01):
        dw1, gradB = hinge_deriv(X, y, w-step*gradW, b)
        gradW, db1 = hinge_deriv(X, y, w, b-step*gradB)
        step = stepSize(X, y, w, b, step, gradW, gradB)
        w = np.subtract(w, step * gradW)
        b -= step * gradB

        gradW, gradB = hinge_deriv(X, y, w, b)
        gradWNext, gradBNext = hinge_deriv(X, y, w-step*gradW, b-step*gradB)


        #gradWNext, gradB = hinge_deriv(X, y, w-step*gradW, b)
        #gradW, gradBNext = hinge_deriv(X, y, w, b-step*gradB)

    #print("w: ", w)
    #print("b: ", b)
    return w, b


#Main
#------------------------------------------------------------------------------

"""
Attempt 1: Fitting to linear data, showing success of GA with varying
numbers of generations
"""

inputAthensMaxMin = pd.read_excel('athens-data.xlsx', usecols="H, I")
dataAthens = np.array(inputAthensMaxMin).T
X = dataAthens[1]
Y = dataAthens[0]
plt.plot(X, Y, '.', label='Temps')


numGens = [100, 1000, 10000]
i = 0
while (i < 3):
    best = runGA(X, Y, numGens[i], "linear", 400)
    m = best["m"]
    b = best["b"]
    plt.plot(X, m*X+b, linewidth = 2.0, label = 'Generations: {}'.format(numGens[i]))
    i += 1

plt.xlabel('Min Temp (C)')
plt.ylabel('Max Temp (C)')
plt.title('Temperature in Athens 1944-1945 w/ Genetic Algorithm')
plt.legend(loc = 'upper left')
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()


"""
Attempt 2: Fitting to SVM test data, showing success of GA vs SVM
"""

inputSVM = pd.read_excel('svm_test_2.xlsx', usecols="A, B, C")
data = np.array(inputSVM)

X = data[:, :2]
y = data[:, 2]

#get line from GA
popSize = 1000
idealFitVal = -10000
best = runGA(X, y, popSize, "classify", idealFitVal)
m = best["m"]
b = best["b"]
plt.plot(data[:, 0], m*data[:, 0]+b, linewidth = 2.0, label = 'Genetic Algorithm')

#get line from SVM
w, b = gradSVM(X, y, np.array([-10, 10]), 10)
m = w.T[0]
plt.plot(data[:, 0], m*data[:, 0]+b, linewidth = 2.0, label='SVM')

plt.plot(data[0:200, 0], data[0:200, 1], '.',) #pos; below SVM line
plt.plot(data[201:, 0], data[201:, 1], 'x',) #neg; above SVM line

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('SVM test data')
plt.legend(loc = 'upper left')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()


"""
Attempt 3: Fitting to pulsar data, showing "success" of GA vs SVM
This may take a while as even a "small" section of the pulsar dataset
is very hard to sort!
"""
inputPulsar = pd.read_excel('pulsar_data.xlsx')
pData = np.array(inputPulsar)
pData = pData[:700, :]

currEps = 500

dists = dist_matrix(pData[:, :8])
edges = edge_matrix(dists, currEps)
iso, removed = isomap(dists, edges, 2)

normal = []
pulsar = []
j = 0
while (j < len(iso)):
    if (j not in removed):
        if (pData[j, 8] ==  0):
            normal.append(iso[j])
        else:
            pulsar.append(iso[j])
    j += 1
nor = np.asarray(normal)
pul = np.asarray(pulsar)

X = iso
y = pData[:, 8]

#get line from GA
popSize = 10000
idealFitVal = -7000
best = runGA(X, y, popSize, "classify", idealFitVal)
m = best["m"]
b = best["b"]
plt.plot(data[:, 0], m*data[:, 0]+b, linewidth = 2.0, label = 'Genetic Algorithm')

#get line from SVM
w, b = gradSVM(X, y, np.array([-10, 10]), 10)
m = w.T[0]
plt.plot(data[:, 0], m*data[:, 0]+b, linewidth = 2.0, label='SVM')


plt.plot(nor[:, 0], nor[:, 1], '.', label='not pulsars')
plt.plot(pul[:, 0], pul[:, 1], '.', label='pulsars')
plt.title('Pulsar data pre-processed w/ iso, eps = {}'.format(currEps))
plt.legend(loc = 'lower left')

plt.show()
