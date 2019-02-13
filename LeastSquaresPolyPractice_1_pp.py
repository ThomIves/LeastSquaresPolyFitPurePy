import LinearAlgebraPurePython as la 
import MachineLearningPurePy as ml

import matplotlib.pyplot as plt


X = [2,3,4]
def y_of_x(x):
    return 0.2*x + 0.5*x**2

Y = []
for x in X:
    Y.append(y_of_x(x))

print('Calculated Y Values:', Y)
print()

plt.scatter(X, Y)


poly_pp = ml.Poly_Features_Pure_Py(order=2)
Xpp = poly_pp.fit_transform(la.transpose([X]))

ls_pp = ml.Least_Squares(add_ones_column=False)
ls_pp.fit(Xpp, Y)
print('LS coefficients:', ls_pp.coefs)

XLS = [0,1,2,3,4,5]
XLSpp = poly_pp.transform(la.transpose([XLS]))

YLS = ls_pp.predict(XLSpp)

plt.plot(XLS, la.transpose(YLS)[0])
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Pure Python Least Squares Line Fit')
plt.show()
