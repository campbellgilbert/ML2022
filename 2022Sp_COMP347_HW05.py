# Name: Campbell Gilbert
# COMP 347 - Machine Learning
# HW No. 5

#*********************
# PROBLEMS ARE POSTED BELOW THE FUNCTIONS SECTION!!!
#*********************

# Libraries
#------------------------------------------------------------------------------
import numpy as np
import scipy.linalg as LA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.image  as mpimg
import pandas            as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy import misc
from itertools import chain


# Functions
#------------------------------------------------------------------------------
def confidence_ellipse(x, y, ax, n_std=3.0, edgecolor='red', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        edgecolor=edgecolor,
        facecolor='none',
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Euclidean edge distances
def dist_matrix(Data, type):
    """Returns the Euclidean distance matrix dists."""
    N = len(Data)
    distMat = np.zeros([N, N])
    i = 0
    while (i < N):
        j = 0
        while (j < N):
            if (i != j):
                if (type == 'MDS'):
                    distMat[i, j] = LA.norm(Data[i] - Data[j])**2
                else:
                    distMat[i, j] = LA.norm(Data[i] - Data[j])
            j += 1
        i += 1
    return distMat

# Multidimensional scaling alg
def classical_mds(Dists,dim):
    """Takes the distance matrix Dists and dimensionality dim of the desired
        output and returns the classical MDS compression of the data."""
    N = np.shape(Dists)[0]
    ones = np.ones((N,N))
    H = np.identity(N) - (1/N)*ones
    XXT = (-1/2)*np.dot(H, np.dot(Dists * Dists, H))
    P, D, PT = LA.svd(XXT)
    Y = np.dot(P, np.diag(np.sqrt(D)))
    return Y[:,:dim]

# Construct edge matrix based on KNN or epsilon ball
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

# Construct ISOMAP graph by replacing Euclidean distances with graph distances
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


# Problem #1 - PCA of Data and SVD Compression
#------------------------------------------------------------------------------

#   1a. Perform principle component analysis on both the random data and sinusoidal
#       data sets.  For each data set plot the corresponding covariance ellipses
#       along with the principle vectors which should be the principle axes of
#       of the ellipses.  Use the confidence_ellipse function given above using values
#       of n_std = 1,2 as the standard deviations.  Some notes: the length of the
#       of the principle axes are given by the standard deviations of the univariate
#       data, hence the square-root of the top-left and bottom-right entries in
#       the covariance matrix.  Additionally, these vectors are orthogonal but will
#       only appear perpendicular if the image is plotted with equal axes; they may
#       appear skew otherwise.  Annotate the images appropriately.

def XTilde(X):
    N = len(X)
    eye = np.identity(N)
    onesVec = np.ones((N, 1))
    oneOneT = 1/N * np.dot(onesVec, onesVec.T)
    return np.dot(np.subtract(eye, oneOneT), X)

def getCovMat(X):
    return (1/(len(X) - 1)) * np.dot(XTilde(X.T), XTilde(X))


inputRand = pd.read_excel('rand_data.xlsx', usecols="A, B")
dataRand = np.array(inputRand)

inputSin = pd.read_excel('sin_data.xlsx', usecols="A, B")
dataSin = np.array(inputSin)

fig, ax = plt.subplots(1, 2)

ranvals, ranvecs = LA.eigh(np.cov(dataRand.T))
sinvals, sinvecs = LA.eigh(np.cov(dataSin.T))

ranvecs[:, 0] =  ranvecs[:, 0] * np.sqrt(ranvals[0])
ranvecs[:, 1] =  ranvecs[:, 1] * np.sqrt(ranvals[1])
sinvecs[:, 0] =  sinvecs[:, 0] * np.sqrt(sinvals[0])
sinvecs[:, 1] =  sinvecs[:, 1] * np.sqrt(sinvals[1])

N1 = np.shape(dataRand)[0]
mu1 = dataRand.sum(axis=0)/N1

N2 = np.shape(dataSin)[0]
mu2 = dataSin.sum(axis=0)/N2
#PCA(dataRand)
#print(XTilde(dataRand))

ax[0].plot(dataRand[:, 0], dataRand[:, 1], '.', label='Random data')
confidence_ellipse(dataRand[:, 0], dataRand[:, 1], ax[0], 1, 'red')
confidence_ellipse(dataRand[:, 0], dataRand[:, 1], ax[0], 2, 'green')
ax[0].arrow(mu1[0], mu1[1], ranvecs[0, 0], ranvecs[1, 0])
ax[0].arrow(mu1[0], mu1[1], ranvecs[0, 1], ranvecs[1, 1])


ax[1].plot(dataSin[:, 0], dataSin[:, 1], '.', label='Sinusoidal data')
confidence_ellipse(dataSin[:, 0], dataSin[:, 1], ax[1], 1, 'red')
confidence_ellipse(dataSin[:, 0], dataSin[:, 1], ax[1], 2, 'green')
ax[1].arrow(mu2[0], mu2[1], sinvecs[0, 0], sinvecs[1, 0])
ax[1].arrow(mu2[0], mu2[1], sinvecs[0, 1], sinvecs[1, 1])

plt.show()

#100% DONE
#   1b. Pick your favorite image from your photo library (something school-appropriate,
#       nothing graphic please!) and convert it to gray-scale.  Use the singular value
#       decomposition to produce a sequence of reconstructions by adding back each s.v.
#       back one at a time.  In other words, if the original decomposition is given by
#       USV.T, then for each reconstruction simply replace S with a matrix S_i where
#       i represents the number of sv's added back to the matrix.  In doing this, construct
#       a curve that displays the reconstruction accuracy as a function of the number
#       of singular values included in the reconstruction.  The reconstruction
#       accuracy can be computed as 1 - (LA.norm(Recon - Orig) / LA.norm(Orig)).
#       Annotate your plot with a legend that displays the number of singular values
#       needed to obtain 80%, 85%, 90%, 95%, and 99% accuracy.  Create a graphic that
#       that shows these five reconstructions along with the original.
fig, ax = plt.subplots(2, 3)

img = Image.open("MLpic.jpeg")
imgMat = np.array(list(img.getdata(band=0)), float)
imgMat.shape = (img.size[1], img.size[0])
imgMat = np.matrix(imgMat)

U, sig, V = LA.svd(imgMat)
reconAccuracy = []
accuracy = [80, 85, 90, 95, 99]
reconRange = np.arange(0, 801, 10)
for i in reconRange:
    reconstruction = np.matrix(U[:, :i]) * np.diag(sig[:i]) * np.matrix(V[:i, :])
    reconAccuracy.append(1 - (LA.norm(reconstruction - imgMat) / LA.norm(imgMat)))
    i += 1

reconValues = [20, 35, 55, 90, 185]
reconAccurateVals = []

j = 0
row = 0
while row < 2:
    col = 0
    while col < 3:
        if (row == 0 and col == 0):
            ax[0, 0].imshow(imgMat, cmap = 'gray')
            ax[0, 0].set_title("Original image")
        else:
            i = reconValues[j]
            reconstruction = np.matrix(U[:, :i]) * np.diag(sig[:i]) * np.matrix(V[:i, :])
            reconAccurateVals.append(1 - (LA.norm(reconstruction - imgMat) / LA.norm(imgMat)))
            ax[row, col].imshow(reconstruction, cmap='gray')
            ax[row, col].set_title('{}% Accuracy'.format(accuracy[j]))
            j += 1
        col += 1
    row += 1
plt.show()

plt.plot(reconRange, reconAccuracy, linewidth = 2.0)
i = 0
while i < 5:
    plt.plot(reconValues[i], reconAccurateVals[i], '.', label='{}% accuracy'.format(accuracy[i]))
    i += 1
plt.xlabel("# of single values recovered")
plt.ylabel("ratio of image norms")
plt.title("Reconstruction Accuracy for SV Compression")
plt.legend(loc = 'lower right')
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()


# Problem #2 - MDS of Breast Cancer Data and SVM Modeling with the Hinge Formalism
#------------------------------------------------------------------------------

#   2a. Complete the function dist_matrix above where the ouput should be the
#       NxN Euclidean distance matrix of a data set Data.

#   2b. Complete the function classical_mds that produces the multidimensional
#       scaling compression of a data set.  An important note here is that you
#       are NOT allowed to simply plug in the data set; this has to take the
#       Euclidean distance matrix as input.  The reason for this is that the
#       original algorithm was conceived of as a blind reconstruction: if only
#       given the mutual distances between points could one recover the original
#       data set?  Make sure to follow the derivation as given in the slides to
#       obtain this.

#   2c. Perform the MDS compression of the breast cancer data from UCI repository
#       (i.e. the same data set we've seen in class).  Reproduce the image from
#       the slides and show that there is a natural separation between the benign
#       malignant cases.  Make sure to color/label your data to reflect this.

inputCancer = pd.read_excel('breast_cancer.xlsx')
cancerData = np.array(inputCancer)
cData = cancerData[:, 1:31]

#process data THEN label and THEN graph
dists = dist_matrix(cData[:, 1:], 'MDS')
proc = classical_mds(dists, 2)

malignant = []
benign = []
i = 0
while (i < len(proc)):
    if (cData[i, 0] == 'M'):
        cData[i, 0] = -1
        malignant.append(proc[i])
    else:
        cData[i, 0] = 1
        benign.append(proc[i])
    i += 1

mal = np.asarray(malignant)
ben = np.asarray(benign)

plt.plot(mal[:, 0], mal[:, 1], '.', label='malignant')
plt.plot(ben[:, 0], ben[:, 1], '.', label='benign')
plt.title('BC data pre-processed w/ MDS (d = 2)')
plt.legend(loc = 'lower left')

plt.show()

#   2d. Using your code from HW #4, perform SVM on the new MDS compression of
#       the breast cancer data.  Specifically perform a 10-fold cross validation
#       using the hinge formulation of SVM and plot a bar plot showing the success
#       percentages of the learned models as well as that of the averaged model.
#       Use backtracking line search to aid in finding optimal stepsizes in the
#       gradient descent.  Use a stopping criterion of comparing the relative sizes
#       of successive derivatives of the form
#
#       while 0.99 < LA.norm(D_0)/LA.norm(D_1) < 1.01:

def hinge_loss(X, y, w, b):
    """Here X is assumed to be Nxn where each row is a data point of length n and
    N is the number of data points.  y is the vector of class labels with values
    either +1 or -1.  w is the support vector and b the corresponding bias."""
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
        if (np.dot(y[i], np.dot(w, X[i])+b) < 1):
            gradientWSum = np.add(gradientWSum, w)
            #print("adding w: ", w)
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

procLabeled = np.zeros((len(proc), 3))
i = 0
while (i < len(proc)):
    procLabeled[i, 0] = cData[i, 0]
    procLabeled[i, 1:] = proc[i]
    i += 1

np.random.shuffle(procLabeled)
totalSets = np.array_split(procLabeled, 10)
trainingSets = totalSets[0:9]
testSet = totalSets[9]

#step 3: train models
trainedModels = np.zeros((9, 3))
wStart = np.asarray([1, 1])
bStart = 10

i = 0
while (i < 9):
    X = trainingSets[i][:, 1:]
    y = trainingSets[i][:, 0]

    w, b = gradSVM(X, y, wStart, bStart)

    trainedModels[i, 0] = b
    trainedModels[i, 1:] = w

    #to graph each SVM -- can be commented out for time's sake
    #"""
    plt.plot(mal[:, 0], mal[:, 1], '.', label='malignant')
    plt.plot(ben[:, 0], ben[:, 1], '.', label='benign')
    m = w.T[0]
    plt.plot(X, m*X+b, linewidth=2.0, label='SVM for training group no. {}'.format(i))

    plt.title('BC data pre-processed w/ MDS (d = 2)')
    plt.legend(loc = 'lower left')
    plt.show()
    #"""

    i += 1

X = testSet[:, 1:]
y = testSet[:, 0]
successRates = np.zeros(9)
i = 0
while (i < 9):
    w = trainedModels[i, 1:]
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

# Problem #3 - Isomap of Breast Cancer Data
#------------------------------------------------------------------------------

#   3a. Complete the function edge_matrix which for a given data set produces an
#       NxN matrix of zeros and ones, 0 meaning two points are not nearest neighbors
#       and 1 indicating they are.  Have this matrix constructed using the epsilon-ball
#       method of nearest neighbors where eps is an input variable to this function.

#   3b. Complete the function isomap which produces the isomap compression of a
#       data set given the original Euclidean distance matrix, the desired edge
#       matrix, and the desired number of output dimensions.  The function should
#       output two variables: the compressed data, and a list of indices representing
#       the data points removed as a result of the Floyd-Warshall algorithm.  I
#       highly recommend putting in a print command somewhere within the Floyd-Warshall
#       part of the isomap algorithm since this has cubic complexity and often is the
#       longest part of the procedure.

#   3c. Produce a sequence of isomap embeddings of the breast cancer data using
#       a value of epsilon ranging from 100 to 500 and incrementing in steps of
#       25.  As in the slides, make sure to annotate your images to show the
#       separation between benign and malignant cases.  This part of your homework
#       will take some time to execute because of Floyd-Warshall.  Go grab a coffee
#       while you're waiting for it to terminate and submit your receipt along
#       with your code as proof.  lol, jk on this last part.

fig, ax = plt.subplots(4, 4, figsize=(20, 16))

i = 0
while (i < 16):
    currEps = 125+(25*i)

    dists = dist_matrix(cData[:, 1:], 'ISO')
    edges = edge_matrix(dists, currEps)
    iso, removed = isomap(dists, edges, 2)

    malignant = []
    benign = []
    j = 0
    while (j < len(iso)):
        if (j not in removed):
            if (cData[j][0] == 'M'):
                malignant.append(iso[j])
            else:
                benign.append(iso[j])
        j += 1
    mal = np.asarray(malignant)
    ben = np.asarray(benign)

    r = i // 4
    c = i % 4

    ax[r, c].plot(mal[:, 0], mal[:, 1], '.', label='malignant')
    ax[r, c].plot(ben[:, 0], ben[:, 1], '.', label='benign')
    ax[r, c].set_title('Eps = {0}, removed = {1}'.format(eps[i], len(removed)))
    ax[r, c].legend(loc = 'lower left')

    i += 1
plt.show()
