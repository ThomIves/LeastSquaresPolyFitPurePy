import LinearAlgebraPurePython as la 
# import MachineLearningPurePy as ml
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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


poly_sk = PolynomialFeatures(degree = 2)
Xsk = poly_sk.fit_transform(la.transpose([X]))

ls_sk = LinearRegression()
ls_sk.fit(Xsk, Y)
print('SKLearn LS coefficients:', ls_sk.coef_, '\n')

XLS = [0,1,2,3,4,5]
XLSsk = poly_sk.transform(la.transpose([XLS]))

YLS = ls_sk.predict(XLSsk)

plt.plot(XLS, YLS)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Pure Python Least Squares Line Fit')
plt.show()
