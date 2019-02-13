import LinearAlgebraPurePython as la 
import MachineLearningPurePy as ml

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

import matplotlib.pyplot as plt

import random
import sys


# Fake data preparation
X = [float(x)/25.0 for x in range(400)]

def y_of_x(x):
    return 0.2*x + 0.7*x**2 + random.uniform(-0.2,0.2)

Y = []
for x in X:
    Y.append(y_of_x(x))

plt.scatter(X, Y) # sys.exit()
X = la.transpose([X])
Y = la.transpose([Y])

# Pure Py Fit
poly_pp = ml.Poly_Features_Pure_Py(order=2)
Xpp = poly_pp.fit_transform(X)
ls_pp = ml.Least_Squares(tol=2, add_ones_column=False)
ls_pp.fit(Xpp, Y)

# SKLearn Fit
poly_sk = PolynomialFeatures(degree = 2)
Xsk = poly_sk.fit_transform(X)
ls_sk = LinearRegression()
ls_sk.fit(Xsk, Y)

# Coefficients Comparison
temp_lsp_coefs = sorted(ls_pp.coefs)
rounded_lsp_coefs = [round(x,8)+0 for x in la.transpose(temp_lsp_coefs)[0]]
print('PurePy  LS coefficients:', rounded_lsp_coefs)
print('SKLearn LS coefficients:', ls_sk.coef_, '\n')
print('Coef Deltas:', [rounded_lsp_coefs[i] - ls_sk.coef_[0][i] for i in range(len(rounded_lsp_coefs))])

# Predictions
XLS = [0,1,2,3,4,5]

XLSpp = poly_pp.transform(la.transpose([XLS]))
YLSpp = ls_pp.predict(XLSpp)
YLSpp = la.transpose(YLSpp)[0]

XLSsk = poly_sk.transform(la.transpose([XLS]))
YLSsk = ls_sk.predict(XLSsk)
YLSsk = YLSsk.tolist()
YLSsk = la.transpose(YLSsk)[0]

# Prediction Differences 
YLSpp.sort()
YLSsk.sort()
deltas = [ YLSpp[i] - YLSsk[i] for i in range(len(YLSpp)) ]
print( '\nPrediction Deltas:', deltas , '\n')

# Plotting
plt.plot(XLS, YLSpp, XLS, YLSsk)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Pure Python Least Squares Line Fit')
plt.show()
