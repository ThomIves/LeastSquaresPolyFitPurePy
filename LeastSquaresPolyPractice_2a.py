import LinearAlgebraPurePython as la 
import MachineLearningPurePy as ml

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

import matplotlib.pyplot as plt

import random
import sys


# Section 1: Fake data preparation 
#   and visualization
X = [1.5,2.0,2.5,3.0,3.5,4.0]
def y_of_x(x):
    return 0.2*x + 0.7*x**2 + random.uniform(-1,1)
Y = []
for x in X:
    Y.append(y_of_x(x))

plt.scatter(X, Y)

# Section 2: Get fake data in correct format
X = la.transpose([X])
Y = la.transpose([Y])

# Section 3: Pure Python Tools Fit
poly_pp = ml.Poly_Features_Pure_Py(order = 2)
Xp = poly_pp.fit_transform(X)
print('PP Feature Names:', poly_pp.get_feature_names())
ls_pp = ml.Least_Squares(tol=2, add_ones_column=False)
ls_pp.fit(Xp, Y)
print()

# Section 4: SciKit Learn Fit
poly_sk = PolynomialFeatures(degree = 2)
X_poly = poly_sk.fit_transform(X)
print('SK Feature Names:', poly_sk.get_feature_names())
ls_sk = LinearRegression()
ls_sk.fit(X_poly, Y)
print()

# Section 5: Coefficients Comparison
tmp_ls_pp_coefs = sorted(ls_pp.coefs)
rounded_ls_pp_coefs = [round(x,8)+0 for x in la.transpose(tmp_ls_pp_coefs)[0]]
print('PurePy  LS coefficients:', rounded_ls_pp_coefs)

tmp_ls_sk_coefs = ls_sk.intercept_.tolist() + ls_sk.coef_[0][1:].tolist()
tmp_ls_sk_coefs = sorted(tmp_ls_sk_coefs)
rounded_ls_sk_coefs = [round(x,8)+0 for x in tmp_ls_sk_coefs]
print('SKLearn LS coefficients:', rounded_ls_sk_coefs, '\n')

print('Coef Deltas:', [rounded_ls_pp_coefs[i] - rounded_ls_sk_coefs[i] 
        for i in range(len(rounded_ls_pp_coefs))])

# Section 6: Create Fake Test Data
XLS = [0,1,2,3,4,5]

# Section 6.1: Predict with Fake Test Data
#   Using Pure Python Tools
XLSpp = poly_pp.transform(la.transpose([XLS]))
YLSpp = ls_pp.predict(XLSpp)
YLSpp = la.transpose(YLSpp)[0]

# Section 6.2: Predict with Fake Test Data
#   Using SciKit Learn Tools
XLSsk = poly_sk.transform(la.transpose([XLS]))
YLSsk = ls_sk.predict(XLSsk)
YLSsk = YLSsk.tolist()
YLSsk = la.transpose(YLSsk)[0]

# Section 7: Calculate Prediction Differences 
YLSpp.sort()
YLSsk.sort()
deltas = [ YLSpp[i] - YLSsk[i] for i in range(len(YLSpp)) ]
print( '\nPrediction Deltas:', deltas , '\n')

# Section 8: Plot Both Methods
plt.plot(XLS, YLSpp, XLS, YLSsk)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Pure Python Least Squares Line Fit')
plt.show()
