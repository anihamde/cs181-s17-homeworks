# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
import random as rando
import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from Perceptron import Perceptron

# Implement this class
class KernelPerceptron(Perceptron):
	def __init__(self, numsamples):
		self.numsamples = numsamples


	# Implement this!
	def fit(self, X, Y):
		self.X = X
		self.Y = Y

		S = []
		alpha = []
		b = 0

		#num = rando.randint(0,len(X)-1)

		for num in range(0,len(Y)):
			ypred = 0
			for i in range(0,len(S)):
				ypred += alpha[i]*np.dot(X[num,:],X[S[i],:])+b

			if(Y[num]*ypred <= 0):
				S.append(num)
				alpha.append(Y[num])

		self.Xold = X
		self.S = S
		self.alpha = alpha
		print("NUMBER OF SVMs IN KERNEL PERC ",len(S))

		# print(len(S))

	# Implement this!
	def predict(self, X):
		self.X = X
		Xold = self.Xold

		S = self.S
		alpha = self.alpha
		b = 0

		Ypredvec = []

		for num in range(0,len(X)):
			ypred = 0
			for i in range(0,len(S)):
				ypred += alpha[i]*np.dot(X[num,:],Xold[S[i],:])+b

			Ypredvec.append(ypred)

		Ytfvals = (np.array(Ypredvec)[:] > 0)

		self.Ypredvec = Ypredvec

		# print(alpha)
		# print("HEYAPO")
		# print(Ypredvec)

		return(Ytfvals)

	def accuracy(self, Y):
		Ypredvec = self.Ypredvec
		Ypredvec = np.array(Ypredvec)
		Yhammond = np.multiply(Ypredvec,Y)
		print(Yhammond)
		count = 0
		for i in Yhammond:
			if(i >= 0):
				count = count + 1
		print(count/len(Ypredvec))
		#print(sum(Ytfvals*(Y+1))/len(Ytfvals*(Y+1)))

# Implement this class
class BudgetKernelPerceptron(Perceptron):
	def __init__(self, beta, N, numsamples):
		self.beta = beta
		self.N = N
		self.numsamples = numsamples
		
	# Implement this!
	def fit(self, X, Y):
		self.X = X
		self.Y = Y

		S = np.array(X[0,:])
		alpha = np.array(Y[0])
		ypredvec = []
		b = 0
		defx = []

		for num in range(1,len(Y)):
			ypred = 0
			ypred += np.dot(alpha,np.dot(S,(X[num,:]).transpose()))

			ypredvec = np.append(ypredvec, ypred)

			defx = np.append(defx,ypred*Y[num])

			if ypred*Y[num] <= self.beta:
				alpha = np.append(alpha, Y[num])
				S = np.vstack((S, X[num,:]))

				if S.shape[0] > N:
					alpha, S = self.remove(alpha, S)

		self.Xold = X
		self.S = S
		self.alpha = alpha

		print("Number of SVMs: ", S.shape[0])

	def remove(self, alpha, S):
		Y = self.Y
		Snew = np.multiply(S, S)
		S2 = np.array(Snew[:,0] + Snew[:,1])

		vecimport = np.multiply(alpha,(np.dot(alpha,np.dot(S,S.transpose()))-np.multiply(alpha,S2)))

		indeximport = np.argmax(vecimport)

		return(np.delete(alpha, indeximport), np.delete(S, indeximport, axis=0))


		


		# for num in range(0,len(Y)):
		# 	ypred = 0
		# 	for i in range(0,len(S)):
		# 		ypred += alpha[i]*np.dot(X[num,:],X[S[i],:])+b

		# 	if(Y[num]*ypred <= beta):
		# 		S.append(num)
		# 		alpha.append(Y[num])
		# 		ypredvec.append(ypred)

		# 	if(len(S) > N):
		# 		S.pop(
		# 			np.argmax(Y[S[i]]*(ypredvec[S[i]]-alpha[i]*np.dot(X[S[i]:],X[S[i],:])))
		# 			)

		# 		ypred

		# 		np.matmul(alpha,np.matmul(X[S,],X.T))

		# 		for num in range(0,len(Y)):
		# 			ypred = 0
		# 			for i in range(0,len(S)):
		# 				ypred += alpha[i]*np.dot(X[num,:],X[S[i],:])+b



		# self.Xold = X
		# self.S = S
		# self.alpha = alpha

		# print(len(S))

	# Implement this!
	def predict(self, X):
		self.X = X
		Xold = self.Xold

		S = self.S
		alpha = self.alpha
		b = 0

		Ypredvec = np.dot(alpha,np.dot(S,X.transpose()))
		# print(Ypredvec)

		# for num in range(0,len(X)):
		# 	ypred = 0
		# 	for i in range(0,len(S)):
		# 		ypred += alpha[i]*np.dot(X[num,:],Xold[S[i],:])+b

		# 	Ypredvec.append(ypred)

		Ytfvals = (Ypredvec[:] > 0)# Ytfvals = (np.array(Ypredvec)[:] > 0)

		self.Ypredvec = Ypredvec

		# print(alpha)
		# print("HEYAPO")
		# print(Ypredvec)

		return(Ytfvals)

	def accuracy(self, Y):
		Ypredvec = self.Ypredvec
		Ypredvec = np.array(Ypredvec)
		Yhammond = np.multiply(Ypredvec,Y)
		print(Yhammond)
		count = 0
		for i in Yhammond:
			if(i >= 0):
				count = count + 1
		print(count/len(Ypredvec))
		#print(sum(Ytfvals*(Y+1))/len(Ytfvals*(Y+1)))



# Implement this class
class SMO(Perceptron):
	def __init__(self, numsamples,C,tau):
		self.numsamples = numsamples
		self.C = C
		self.tau = tau

	def find_tolerance(self,tau,alpha):
		X = self.X
		A = self.A
		B = self.B
		g = self.g
		counter = 0
		indicator = 0
		i, j = 0, 0
		while(indicator == 0):
			counter += 1
			if alpha[i] < B[i] and alpha[j] > A[j] and (g[i] - g[j]) > tau:
				indicator = 1
				self.i = i
				self.j = j
			else:
				i += 1

				if(i == len(X)):
					i = 0
					j += 1

					if(j == len(X)):
						j = 0
						indicator = 2

		self.indicator = indicator

		print(g[i])
		print(g[j])
		return(i, j)

	# Implement this!
	def fit(self, X, Y):
		self.X = X
		self.Y = Y

		print(0)

		b = 0

		A = []
		B = []

		print(1)

		for i in range(len(X)):
			A.append(min(0,C*Y[i]))
			B.append(max(0,C*Y[i]))

		self.A = A
		self.B = B

		print(A)
		print(B)

		print(2)

		alpha = np.repeat(0,len(X)) 

		print(3)

		g = []

		for i in range(len(X)):
			pmat = np.dot(X[i,:],X.transpose())
			pmat = np.sum(pmat)
			g.append(Y[i] - alpha[i]*pmat)
			#print(i/len(X))

		self.g = g
		#g = np.subtract(Y, np.dot(alpha, np.dot(X,X.transpose())))

		print(4)

		indicator = 0
		i, j = self.find_tolerance(tau,alpha)
		print(i,j)
		indicator = self.indicator


		while(indicator != 2):
			lambdah = min((g[i]-g[j])/(np.dot(X[i],X[i])+np.dot(X[j],X[j])-2*np.dot(X[i],X[j])),B[i]-alpha[i],alpha[j]-A[j])
			print(i, g[i])
			print(j, g[j])
			self.lambdah = lambdah
			alpha[i] = alpha[i] + lambdah
			alpha[j] = alpha[j] - lambdah
			self.alpha = alpha
			
			g = g - lambdah*(np.dot(np.subtract(X[i],X[j]),X.transpose()))	

			i, j = self.find_tolerance(tau,alpha)
			indicator = self.indicator

		print(5)

		self.Xold = X
		self.alpha = alpha
		self.lambdah = lambdah
		self.g = g
		self.A = A
		self.B = B

		# print(len(S))




	# Implement this!
	def predict(self, X):
		self.X = X
		Xold = self.Xold

		S = self.S
		alpha = self.alpha
		b = 0

		Ypredvec = np.dot(alpha,np.dot(S,X.transpose()))

		# for num in range(0,len(X)):
		# 	ypred = 0
		# 	for i in range(0,len(S)):
		# 		ypred += alpha[i]*np.dot(X[num,:],Xold[S[i],:])+b

		# 	Ypredvec.append(ypred)

		Ytfvals = (Ypredvec[:] > 0)# Ytfvals = (np.array(Ypredvec)[:] > 0)

		self.Ypredvec = Ypredvec

		# print(alpha)
		# print("HEYAPO")
		# print(Ypredvec)

		return(Ytfvals)

	def accuracy(self, Y):
		Ypredvec = self.Ypredvec
		Ypredvec = np.array(Ypredvec)
		count = 0
		for i in Ypredvec:
			if(Ypredvec[i]*Y[i]>=0):
				count = count + 1
		print(count/len(Ypredvec))
		#print(sum(Ytfvals*(Y+1))/len(Ytfvals*(Y+1)))


# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:4000, :2]
Y = data[:4000, 2]

val = np.loadtxt("val.csv", delimiter=',')
Xval = val[:, :2]
Yval = val[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0.344
N = 150
numsamples = 20000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
# k = KernelPerceptron(numsamples)
# startime = time.time()
# k.fit(X,Y)
# print("KERNEL PERCEPTRON TOOK ",time.time()-startime," seconds!")
# k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)#True)

# k.predict(X)
# k.accuracy(Y)

# k.predict(Xval)
# k.accuracy(Yval)

# bk = BudgetKernelPerceptron(beta, N, numsamples)
# starttime = time.time()
# bk.fit(X, Y)
# print("This took ",time.time()-starttime," seconds!")
# bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)#True)

# bk.predict(X)
# bk.accuracy(Y)

# print("HOLD UP!!!")

# bk.predict(Xval)
# bk.accuracy(Yval)

# clf = svm.SVC()
# starttime = time.time()
# clf.fit(X, Y)
# print("This took ",time.time()-starttime," seconds!")
# print("Naive SVM from sklearn accuracy is ",accuracy_score(Yval,clf.predict(Xval)),".")

# X = X
# Y = Y
# width = 0

# # Create a grid of points
# x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
# y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
# xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
#     y_max, .05))

# # Flatten the grid so the values match spec for self.predict
# xx_flat = xx.flatten()
# yy_flat = yy.flatten()
# X_topredict = np.vstack((xx_flat,yy_flat)).T

# # Get the class predictions
# Y_hat = clf.predict(X_topredict)

# Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))

# cMap = c.ListedColormap(['r','b','g'])

# # Visualize them.
# plt.figure()
# plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
# plt.savefig('svm.png')
# plt.show()






C = 0.1
tau = 0.1

smomodel = SMO(numsamples,C,tau)
smomodel.fit(X,Y)
#smomodel.visualize("smomodel.png", width=0, show_charts=True, save_fig=True, include_points=False)






