# Name: Campbell Gilbert
# COMP 347 - Machine Learning
# HW No. 2

#you will have to fullscreen the graphs at the end
#sorry for the small plots but it was that or showing 20 of almost the same graph!
#& thank you for the extra time to finish :)

# Libraries
#------------------------------------------------------------------------------
import numpy             as np
import scipy.linalg      as LA
import time
import pandas            as pd
import matplotlib.pyplot as plt
import math


# Problem 1 - Linear Regression with Athens Temperature Data
#------------------------------------------------------------------------------

# In the following problem, implement the solution to the least squares problem.

# 1a. Complete the following functions:

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


def LLS_Solve(x,y, deg):
    """Find the vector w that solves the least squares regression.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit."""

    A = A_mat(x, deg)
    ATAinv = np.linalg.inv(np.dot(A.T, A))
    w = np.dot(ATAinv, np.dot(A.T, y))
    return w

def LLS_ridge(x,y,deg,lam):
    """Find the vector w that solves the ridge regresssion problem.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.
       lam: parameter for the ridge regression."""
    A = A_mat(x, deg)
    ATA = np.dot(A.T, A)
    ATAPlusLam = np.add(ATA, lam*np.identity(deg + 1))
    ATALamInv = np.linalg.inv(ATAPlusLam)
    ATy = np.dot(A.T, y)
    w = np.dot(ATALamInv, ATy)
    return w


def poly_func(data, coeffs):
    """Produce the vector of output data for a polynomial.
       data: x-values of the polynomial.
       coeffs: vector of coefficients for the polynomial."""
    A = A_mat(data, len(coeffs) - 1)
    return np.dot(A, coeffs)


def LLS_func(x,y,w,deg):
    """The linear least squares objective function.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights.
       deg: degree of the polynomial."""
    A = A_mat(x, deg)
    AwMinusY = np.subtract(np.dot(A, w), y)
    f = np.dot(AwMinusY, AwMinusY)
    return f


def RMSE(x,y,w):
    """Compute the root mean square error.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights."""
    A = A_mat(x, len(w) - 1)
    temp = LA.norm(np.subtract(y, np.dot(A, w)))
    temp2 = 1/len(x)
    rmse = math.sqrt(temp2*(temp**2))
    return rmse


# 1b. Solve the least squares linear regression problem for the Athens
#     temperature data.  Make sure to annotate the plot with the RMSE.

inputAthens = pd.read_excel('athens-data.xlsx', usecols="B, J")
degree = 6
dataAthens = np.array(inputAthens).T

w = LLS_Solve(dataAthens[0], dataAthens[1], degree)
R = RMSE(dataAthens[0], dataAthens[1], w)
plt.plot(dataAthens[0], dataAthens[1], '.')
plt.plot(dataAthens[0], poly_func(dataAthens[0], w), linewidth=2.0)
plt.xlabel('Days from November 26, 1944\nRMSE: {}' .format(R))
plt.ylabel('Temperature Celcius')
plt.title('Temperature in Athens 1944-1945 fitted at degree {}' .format(degree))
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.0)
plt.show()


# Problem 2 -- Polynomial Regression with the Yosemite Visitor Data
#------------------------------------------------------------------------------

# 2a. Create degree-n polynomial fits for 5 years of the Yosemite data, with
#     n ranging from 1 to 20.  Additionally, create plots comparing the
#     training error and RMSE for 3 years of data selected at random (distinct
#     from the years used for training).

inputYosemite = pd.read_excel('yosemite_visits.xlsx', usecols=range(1, 13))
dataYosemite = np.array(inputYosemite)

i = 0
x = np.zeros(5*12)
y = np.zeros(5*12)
years = np.zeros(5)
currMonth = 0
while (currMonth < 12):
    currYear = 0
    while (currYear < 5):
        x[(currMonth * 5) + currYear] = currMonth + 1
        y[(currMonth * 5) + currYear] = dataYosemite[(39 - currYear), currMonth]
        years[currYear] = 1979 + currYear
        currYear += 1
    currMonth += 1

xTest = np.zeros(3*12)
yTest = np.zeros(3*12)
yearsTest = np.zeros(3)
currMonth = 0

while (currMonth < 12):
    currYear = 0
    while (currYear < 3):
        xTest[(currMonth * 3) + currYear] = currMonth + 1
        yTest[(currMonth * 3) + currYear] = dataYosemite[(currYear), currMonth]
        yearsTest[currYear] = 2018 - currYear
        currYear += 1
    currMonth += 1

# Create degree-n polynomial fits for 5 years of the Yosemite data, with
#     n ranging from 1 to 20.
fig, axs = plt.subplots(4, 5, constrained_layout=True)
row = 0
deg = 1
while (row < 4):
    col = 0
    while (col < 5):
        i = 0
        while (i < 5):
            axs[row, col].plot(x[i:(i+55):5], y[i:(i+55):5], '-o', label = 'Year {}'.format(years[i]))
            i += 1
        axs[row, col].plot(x, poly_func(x, LLS_Solve(x, y, deg)), label  = 'deg {}'.format(deg))
        axs[row, col].set_title('Yosemite visitors by month for deg {}'.format(deg))
        axs[row, col].set_xlabel('Months (Starting from Jan)')
        axs[row, col].set_ylabel('Visitors')
        if (row == 0 & col == 0):
            axs[row, col].legend(loc='lower left')
        deg += 1
        col += 1
    row += 1
plt.show()


# Additionally, create plots comparing the
#     training error and RMSE for 3 years of data selected at random (distinct
#     from the years used for training).

#finding RMSE, training error for test year
figs, axs = plt.subplots(3, 1, constrained_layout=True)
n = 0
while (n < 3):
    rmses = np.zeros(20)
    error = np.zeros(20)
    degs = np.zeros(20)
    i = 0
    while (i < 20):
        weights = LLS_Solve(xTest[n:(n + 33):3], yTest[n:(n + 33):3], i + 1)
        error[i] = np.sqrt((1/60) * LLS_func(xTest[n:(n + 33):3], yTest[n:(n + 33):3], weights, i + 1))
        rmses[i] = RMSE(xTest, yTest, weights)
        degs[i] = i + 1
        i += 1

    axs[n].plot(degs, rmses, '-o', label = 'RMSEs')
    axs[n].plot(degs, error, '-o', label = 'Training Error')

    axs[n].set_title('Training vs. RMSE for year {}'.format(yearsTest[n]))
    axs[n].set_xlabel('Polynomial Degree')
    axs[n].set_ylabel('RMSE')
    axs[n].legend(loc='upper left')
    n += 1
plt.show()

# 2b. Solve the ridge regression regularization fitting for 5 years of data for
#     a fixed degree n >= 10.  Vary the parameter lam over 20 equally-spaced
#     values from 0 to 1.  Annotate the plots with this value.
fig, axs = plt.subplots(4, 5, constrained_layout=True)
row = 0
deg = 8
lam = np.arange(0, 1, 0.05)
currLam = 0
ridge = np.ndarray(shape=(20, 12))
while (row < 4):
    col = 0
    while (col < 5):
        #1: plot visitors per year
        i = 0
        while (i < 5):
            axs[row, col].plot(x[i:(i+55):5], y[i:(i+55):5], label = 'Year {}'.format(years[i]))
            i += 1
        axs[row, col].plot(x, poly_func(x, LLS_ridge(x, y, deg, lam[currLam])), '-X', label  = 'lam {}'.format(lam[currLam]))
        axs[row, col].set_title('Yosemite RR deg 8 lam {}'.format(lam[currLam]))
        axs[row, col].set_xlabel('Months (Starting from Jan)')
        axs[row, col].set_ylabel('Visitors')
        if (row == 0 & col == 0):
            axs[row, col].legend(loc='lower left')
        currLam += 1
        col += 1
    row += 1
plt.show()
