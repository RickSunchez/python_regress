import numpy as np
import matplotlib.pyplot as plt
# POLYNOMINAL
# DOCS: https://www.sklearn.org/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

file = open("data.txt", "r")
data = file.read().split("\n")

data_x = []
data_y = []

for i in range(len(data)):
	data_x.append(i)
	data_y.append(float(data[i]))

x = np.array(data_x).reshape((-1, 1))
y = np.array(data_y)

plt.scatter(x, y, color="blue")

poly = make_pipeline(PolynomialFeatures(degree=3), Ridge())
poly.fit(x,y)

y_poly = poly.predict(x)

plt.plot(x, y_poly, color="red")

pred_x = np.array( [len(x)] ).reshape((-1, 1))
pred_y = poly.predict(pred_x)

plt.scatter(pred_x, pred_y, color="red")

plt.show()
