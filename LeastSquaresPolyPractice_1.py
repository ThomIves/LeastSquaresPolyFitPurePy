import LinearAlgebraPurePython as la 
import MachineLearningPurePy as ml

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import sys


# Section 1: Fake data preparation 
#   and visualization
X = [2,3,4]
def y_of_x(x):
    return 0.2*x + 0.5*x**2
Y = []
for x in X:
    Y.append(y_of_x(x))

print('Calculated Y Values:', Y, '\n')

plt.scatter(X, Y)

# Section 2: Get fake data in correct format
X = la.transpose([X])
Y = la.transpose([Y])

# Section 3: Pure Python Tools Fit
poly_pp = ml.Poly_Features_Pure_Py(order = 2)
Xpp = poly_pp.fit_transform(X)
ls_pp = ml.Least_Squares(add_ones_column=False)
ls_pp.fit(Xpp, Y)

# Section 4: SciKit Learn Fit
poly_sk = PolynomialFeatures(degree = 2)
Xsk = poly_sk.fit_transform(X)
ls_sk = LinearRegression()
ls_sk.fit(Xsk, Y)

# Section 5: Coefficients Comparison
formatted_lsp_coefs = [round(x,9) 
    for x in la.transpose(ls_pp.coefs)[0]]
print('PurePy  LS coefficients:', formatted_lsp_coefs)
print('SKLearn LS coefficients:', ls_sk.coef_, '\n')

# Section 6: Create Fake Test Data
XLS = [0,1,2,3,4,5]

# Section 6.1: Predict with Fake Test Data
#   Using Pure Python Tools
XLS_pp = poly_pp.transform(la.transpose([XLS]))
YLS_pp = ls_pp.predict(XLS_pp)
YLS_pp = la.transpose(YLS_pp)[0]

# Section 6.2: Predict with Fake Test Data
#   Using SciKit Learn Tools
new_X = poly_sk.transform(la.transpose([XLS]))
YLS_sk = ls_sk.predict(new_X)
YLS_sk = YLS_sk.tolist()
YLS_sk = la.transpose(YLS_sk)[0]

# Section 7: Calculate Prediction Differences 
deltas = [ YLS_pp[i] - YLS_sk[i] 
    for i in range(len(YLS_pp)) ]
print( 'Prediction Deltas:', deltas , '\n')

# Section 8: Plot Both Methods
plt.plot(XLS, YLS_pp, XLS, YLS_sk)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Pure Python Least Squares Line Fit')
plt.show()
