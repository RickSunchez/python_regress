import numpy as np
import matplotlib.pyplot as plt
import random


def genRandPTs(_len_by_range=5, step=1, ranges=[[210,220], [30,50], [110,150]]):
	out_x = []
	out_y = []

	for i in range(len(ranges)):
		r = ranges[i]
		for x in range(_len_by_range):
			out_x.append( (i * _len_by_range) + x )
			out_y.append(random.randint(r[0], r[1]))

	return out_x, out_y

y1 = [28, 54, 80, 89, 115]
x1 = [1, 2, 3, 4, 5]

# docs: http://testent.ru/publ/studenty/vysshaja_matematika/linejnaja_regressija_ispolzovanie_metoda_naimenshikh_kvadratov_mnk/35-1-0-1149

# LINEAR CALC
def linregress(x, y, w=None, b=None):
	# Создаём массивы numpy, для корректной работы библиотеки
	x = np.array(x, dtype=np.float64)
	y = np.array(y, dtype=np.float64)

	if w is None:
		# Создаем массив с длинной равной массиву x, заполненный единицами
		w = np.ones(x.size, dtype=np.float64)

	wxy = np.sum(w*y*x)
	wx  = np.sum(w*x)
	wy  = np.sum(w*y)
	wx2 = np.sum(w*x*x)
	sw  = np.sum(w)

	den = wx2*sw - wx*wx

	if den == 0:
		den = np.finfo(np.float64).eps

	if b is None:
		k = (sw*wxy - wx*wy) / den
		b = (wy - k*wx) / sw
	else:
		k = (wx*y - wx*b) / wx2

	return k, b

# LINEAR SKLEARN
from sklearn.linear_model import LinearRegression

x = np.array(x1).reshape((-1, 1))
y = np.array(y1)

model = LinearRegression()

model.fit(x,y)

k_model = model.coef_[0]
b_model = model.intercept_

k_calc, b_calc = linregress(x1, y1)

# POLYNOMINAL
# DOCS: https://www.sklearn.org/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

mesh_x_clear, mesh_y_clear = genRandPTs()

mesh_x = np.array(mesh_x_clear).reshape((-1, 1))
mesh_y = np.array(mesh_y_clear)

plt.scatter(mesh_x, mesh_y, color="blue")

colors = ["red", "green", "blue", "purple"]
for deg in range(2,6):
	poly = make_pipeline(PolynomialFeatures(degree=deg), Ridge())
	poly.fit(mesh_x,mesh_y)

	y_poly = poly.predict(mesh_x)
	plt.plot(mesh_x, y_poly, color=colors[deg-2])

	pred_x = np.array( [len(mesh_x_clear)+1] ).reshape((-1, 1))
	pred_y = poly.predict(pred_x)

	plt.scatter(pred_x, pred_y, color=colors[deg-2])

plt.show()
