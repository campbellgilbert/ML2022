# Name: Campbell Gilbert
# COMP 347 - Machine Learning
# HW No. 1

import numpy             as np
import scipy.linalg      as LA
import time
import pandas            as pd
import matplotlib.pyplot as plt

"""
# TODO:

- 1d: code to demonstrate structure
"""
# Problem 1 - Some Linear Algebraic Observations
#------------------------------------------------------------------------------
"""
 1a. -- Randomly initialize a matrix A that has fewer columns than rows
 (e.g. 20x5) and compute the eigenvalues of the matrices A^TA and AA^T. What
 do you observe about the eigenvalues of the two matrices?  Compute the
 determinant of the smaller matrix and compare this to the product of all the
 eigenvalues.
 Write a comment explaining your observations.
"""

print("1a")
print()

# Initialize matrix D:
A = np.random.randint(10, size=(6, 4))
print(A)

#find matrix A^T
AT = np.transpose(A)
print(AT)

#matrix multiply: A^TA & AA^T
AAT = np.dot(A, AT)
print(AAT)
ATA = np.dot(AT, A)
print(ATA)

print()
print()

#compute eigenvalues
eigsAAT = np.linalg.eigvals(AAT)
print(eigsAAT)
eigsATA = np.linalg.eigvals(ATA)
print(eigsATA)

print()
print()

#compute det of smaller matrix (ATA)
detATA = LA.det(ATA)
print(detATA)

#compute product of all eigenvalues
prodEigsAAT = np.prod(eigsAAT)
prodEigsATA = np.prod(eigsATA)
prodAllEigs = prodEigsAAT * prodEigsATA
print(prodAllEigs)

#Write a comment explaining your observations.
"""
Comparing determinant of smaller matrix to product of all eigenvalues:
- Product of all eigenvalues is much, much much smaller than detATA
"""

"""
 1b. -- For the smaller matrix above, find the eigenvalues of the
 inverse and compare these to the eigenvalues of the original matrix.
 What is their relationship?  Demonstrate their relationship
 by writing some code below.
 Write a comment explaining the relationship and how your code demonstrates this
"""
print()
print()
print("1b")
print()

#find inverse of ATA
ATAinv = np.linalg.inv(ATA)
print("Inverse of ATA")
print(ATAinv)
print()
print()

#find egnvals of ATA^-1
eigsATAinv = np.linalg.eigvals(ATAinv)
print("egnvals of ATA^-1")
print(eigsATAinv)
print("original egnvals")
print(eigsATA)

invEigsATA = 1/eigsATA
print("inverse of original eigenvalues")
print(invEigsATA)

"""
Relationship between egnvals of ATA-1 and egnvals of ATA:
- the eigenvalues of ATA-1 are the inverses of the eigenvalues of ATA
How code demonstrates:
- 1/eigsATA is shown to equal eigs of ATA^-1
"""

"""
1c. -- Initialize a random, square, non-symmetric matrix C.
Find the eigenvalues of both C and its transpose.
What do you notice about the eigenvalues for both matrices?
Show that they are the same by showing that the sum of their square differences
amounts to a floating point error.
NOTE: you will likely need to sort both arrays.
"""
print()
print()
print("1c")
print()

#initialize random square non-symmetric matrix C
C = np.random.randint(10, size=(4, 4))
print(C)

#find C^T
CT = np.transpose(C)
print(CT)

print()
print()

#find egnvals of C & C^T
eigsC = np.linalg.eigvals(C)
print(eigsC)
eigsCT = np.linalg.eigvals(CT)
print(eigsCT)
print()
print()

#sort both arrays
np.sort(eigsC)
print(eigsC)
np.sort(eigsCT)
print(eigsCT)
"""
What do you notice about the eigenvalues for both matrices?
- The eigenvalues are the exact same
Show that they are the same by showing that the sum of their square differences
amounts to a floating point error.
- the SSD is very very small to the point where it could be a floating point error
"""

#sum of square differences
diff = np.linalg.norm(eigsC-eigsCT)
print("SSD: \n", diff)

"""
1d. -- Finding the eigenvalues and eigenvectors of a matrix is one example of
 a MATRIX DECOMPOSITION.  There are in fact many ways to decompose a matrix
 into a product of other matrices (the factors usually possessing some
 desirable property).  Explore the following matrix decompositions.  Write a
 an explanation for what each one is doing and demonstrate the properties
 of the factors by writing code that demonstrates their structure.  Use the
 internet to find formal and informal articles explaining the various
 decompositions and cite your sources.  Write your ideas in a comment for
 each one below:"""
#writing code that demonstrates their structure
print()
print()
print("1d")
print()

D = np.random.randint(10, size=(6, 4))
print("D: \n", D)

# LU Factorization
# D = LU for lower triangular matrix L and upper triangular matrix U
# https://www.math.ucdavis.edu/~anne/WQ2007/mat67-Ln-LU_Factorization.pdf
# LA.lu(D)
print("\nLU FACTORIZATION:\n")

P, L, U = LA.lu(D)
print("P: \n", P)
print("L: \n", L)
print("U: \n", U)

PLU = (P @ L @ U)
print("D: \n", D)
print("PLU: \n", PLU)


# QR Factorization
# A = QR for A = mxn, Q = mxn, R = nxn; QTQ = I, R is upper triangular & invble
# http://ee263.stanford.edu/lectures/qr.pdf
# LA.qr(D)
print("\nQR FACTORIZATION:\n")


Q, R = LA.qr(D)
print("Q: \n", Q)
print("R: \n", R)
QR = np.dot(Q, R)
print("QR: \n", QR)
print("D: \n", D)
#either the same or very, very, very close!

# Singular Value Decomposition (SVD)
# A = UΣV^T for U orthog, Σ diagonal, V orthog
# https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/positive-definite-matrices-and-applications/singular-value-decomposition/MIT18_06SCF11_Ses3.5sum.pdf
# LA.svd(D)
print("\nSVD FACTORIZATION:\n")

X = np.random.randint(10, size=(6, 6))

U, sigma, VT = LA.svd(X)
print("U: \n", U)
print("sigma: \n", sigma)
print("VT: \n", VT)
USigVT = (U @ np.diag(sigma) @ VT)
#USigVT = U.dot(sigma).dot(VT)
print("USigVT: \n", USigVT)
print("X (original matrix for this problem): \n", X)


#print(SVDD)

# Problem 2 - Run Times and Efficiency
#------------------------------------------------------------------------------
"""It turns out that inverting a matrix using LA.solve
 is actually quite a bit slower than using LA.inv.  How bad is it?  Find out
 for yourself by creating histograms of the run times.  Compare the two matrix
 inversion methods for matrices of sizes 5, 10, 20, 50, and 200 with 1000
 samples for each size.  Record the amount of time each trial takes in two
 separate arrays and plot their histograms (with annotations and title).  You
 can randomly initialize a matrix using any method you wish, but using np.random.rand
 is especially convenient.

 Note: In plotting your histograms use the 'alpha' parameter to adjust the
 color transparency so one can easily see the overlapping parts of your graphs.

 Some references:
 https://scicomp.stackexchange.com/questions/22105/complexity-of-matrix-inversion-in-numpy
 https://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html"""

#Compare the two matrix inversion methods for matrices of sizes
#5, 10, 20, 50, and 200 with 1000 samples for each size.

#time since 1/1/1970

#methods

sizes = np.array([5, 10, 20, 50, 200])
invTime = np.zeros((5, 1000))
solTime = np.zeros((5, 1000))
#for: LA.inv
j = 0
i = 0
while j < 5:
    while i < 1000:
        rando = np.random.randint(10, size = (sizes[j], sizes[j]))
        eye = np.eye(sizes[j], dtype=int)

        start = time.time_ns()
        LA.inv(rando)
        end = time.time_ns()
        invTime[j, i] = (end - start)

        start1 = time.time_ns()
        LA.solve(rando, eye)
        end1 = time.time_ns()
        solTime[j, i] = (end1 - start1)

        i+=1
    i = 0
    j+=1
print("Inv time: \n", invTime)
print("Solve time: \n", solTime)

print("The invert time function is much faster and is much more consistent.")
i = 1
min = np.array([5000, 7000, 15000, 40000, 400000])
max = np.array([35000, 35000, 50000, 130000, 1500000])
while i < 6:
    plt.subplot(2, 3, i)
    plt.hist(solTime[i-1], bins='auto', color='red', label='Solve Time')
    plt.hist(invTime[i-1], bins='auto', color='blue', label='Inverse Time')
    plt.title('Inv Time vs Solve Time for Size {} Matrices'.format(sizes[i-1]))
    #plt.gca().set_xlim(min[i-1], max[i-1])
    i += 1
    plt.legend(loc='upper right')
    plt.xlabel('Time taken in nanoseconds')
    plt.ylabel('Number of results')
plt.gcf().set_size_inches(plt.gcf().get_size_inches()*3)
plt.show()
plt.cla()
