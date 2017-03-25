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

		# print(g[i])
		# print(g[j])
		return(i, j)

	# Implement this!
	def fit(self, X, Y):
		self.X = X
		self.Y = Y

		#print(0)

		b = 0

		A = []
		B = []

		#print(1)

		for i in range(len(X)):
			A.append(min(0,C*Y[i]))
			B.append(max(0,C*Y[i]))

		self.A = A
		self.B = B

		#print(A)
		#print(B)

		#print(2)

		alpha = np.repeat(0,len(X))

		#print(3)

		g = []

		for i in range(len(X)):
			pmat = np.dot(X[i,:],X.transpose())
			pmat = np.sum(pmat)
			g.append(Y[i] - alpha[i]*pmat)
			#print(i/len(X))

		self.g = g
		#g = np.subtract(Y, np.dot(alpha, np.dot(X,X.transpose())))

		#print(4)

		indicator = 0
		i, j = self.find_tolerance(tau,alpha)
		#print(i,j)
		indicator = self.indicator

		while(indicator != 2):
			lambdah = min((g[i]-g[j])/(np.dot(X[i],X[i])+np.dot(X[j],X[j])-2*np.dot(X[i],X[j])),B[i]-alpha[i],alpha[j]-A[j])
			# print(i, g[i]-g[j])
			self.lambdah = lambdah
			alpha[i] = alpha[i] + lambdah
			alpha[j] = alpha[j] - lambdah
			self.alpha = alpha
			
			for s in range(len(X)):
				g[s] = g[s] - lambdah*(np.dot(np.subtract(X[i],X[j]),X[s]))

			i, j = self.find_tolerance(tau,alpha)
			indicator = self.indicator
			print(i,j)

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

		#S = self.S
		alpha = self.alpha
		b = 0

		#print(len(X))

		Ypredvec = np.dot(alpha,np.dot(Xold,X.transpose()))

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

class LASVM(Perceptron):
	def __init__(self, numsamples,C,tau,nseed,iter):
		self.numsamples = numsamples
		self.C = C
		self.tau = tau
		self.nseed = nseed
		self.iter = iter

	def argwith(self, g, S, alpha, is_argmax):
		new_gs = []
		for s in S:
			if is_argmax and alpha[s] < self.B[s]:
				new_gs.append(g[s])
			elif not is_argmax and self.A[s] < alpha[s]:
				new_gs.append(g[s])
		if is_argmax:
			return S[np.argmax(new_gs)]
		else: 
			return S[np.argmin(new_gs)]
	
	def PROCESS(self,k,g,S,alpha,X,Y):
		if k not in S:
			alpha[k] = 0
			
			sumstuff = 0
			for s in S:
				sumstuff += alpha[s]*np.dot(X[k,:],X[s,:])
			g[k] = Y[k] - sumstuff
			S.append(k)
			self.alpha = alpha
			self.g = g
			self.S = S
      	
			if Y[k] == 1:
				i = k
				j = self.argwith(g, S, alpha, 0)
			else:
				j = k
				i = self.argwith(g, S, alpha, 1)
        
			if alpha[i] < self.B[i] and alpha[j] > self.A[j] and (g[i] - g[j]) > tau:
				lambdah = min((g[i]-g[j])/(np.dot(X[i],X[i])+np.dot(X[j],X[j])-2*np.dot(X[i],X[j])),self.B[i]-alpha[i],alpha[j]-self.A[j])
				# print(i, g[i]-g[j])
				self.lambdah = lambdah
				alpha[i] = alpha[i] + lambdah
				alpha[j] = alpha[j] - lambdah
				self.alpha = alpha

				for s in range(len(X)):
 					g[s] = g[s] - lambdah*(np.dot(np.subtract(X[i],X[j]),X[s]))
          
				self.g = g
          
				return(0)
			else:
				return(1)
		else:
			return(1)
          
            
	def REPROCESS(self,g,S,alpha,X,Y):
		i = self.argwith(g, S, alpha, 1)
		j = self.argwith(g, S, alpha, 0)
    
		if alpha[i] < self.B[i] and alpha[j] > self.A[j] and (g[i] - g[j]) > tau:
			lambdah = min((g[i]-g[j])/(np.dot(X[i],X[i])+np.dot(X[j],X[j])-2*np.dot(X[i],X[j])),self.B[i]-alpha[i],alpha[j]-self.A[j])
			# print(i, g[i]-g[j])
			self.lambdah = lambdah
			alpha[i] = alpha[i] + lambdah
			alpha[j] = alpha[j] - lambdah
			self.alpha = alpha
        
			for s in range(len(X)):
				g[s] = g[s] - lambdah*(np.dot(np.subtract(X[i],X[j]),X[s]))
        
			self.g = g
      
			i = self.argwith(g, S, alpha, 1)
			j = self.argwith(g, S, alpha, 0)
        
			for s in S:
				if alpha[s] == 0:
					if Y[s] == -1 and g[s] >= g[i]:
						S.remove(s)
					elif Y[s] == 1 and g[s] <= g[j]:
						S.remove(s)
            
			self.S = S
              
			b = (g[i]+g[j])/2
      
			self.b = b
      
			delta = g[i] - g[j]
      
			self.delta = delta

	# Implement this!
	def fit(self, X, Y):
		#X is the data, all 20k points, only 2 columns
		self.X = X
		#Y is the labels, 1 or -1
		self.Y = Y
		#Number of initial vectors (of 1 and -1 each)
		nseed = self.nseed
		#Number of times running through process/reproccess
		iter = self.iter
		#Support vector INDEX set
		S = []
		# Slabs = []
	
		#Bias
		b = 0
	
		self.b = b
		
		#A and B ?? equation 5
		A = []
		B = []

		print(1)

		#Fill A and B based on C from error term
		for i in range(len(X)):
			A.append(min(0,self.C*Y[i]))
			B.append(max(0,self.C*Y[i]))

		self.A = A
		self.B = B
		
		#helper vars
		satisfied1 = 0
		satisfiedneg1 = 0
		i = 0

		print(2)
    
		#Adds nseed +1s and nseed -1s to S
    
		while(satisfied1 < nseed or satisfiedneg1 < nseed):
			if Y[i] == 1 and satisfied1 < nseed:
				S.append(i)
				satisfied1 += 1
			elif Y[i] == -1 and satisfiedneg1 < nseed:
				S.append(i)
				satisfiedneg1 += 1
			i += 1
    
		print(3)

		#initialize Alpha to 0, alpha is 1 x n
		alpha = np.repeat(0, len(X))
	
		g = []
		
		#initialization of g, the gradient (1 x n)
		for i in range(len(X)):
			pmat = np.dot(X[i,:],X.transpose())
			# pmat is 1 x n so we need to sum over this
			pmat = np.sum(pmat)
			# append (y - alpha*pmat) to g
			g.append(Y[i] - alpha[i]*pmat)
		
		self.g = g

		print(4)
	
	
		###STEP 2

		self.lambdah = 0
		self.delta = 0
	
		#Fixed number of calls to process and reprocess
		for q in range(iter):
			randok = rando.randint(0, len(g))
	      
			###PROCESS
			#Bail_out = 1 means it bailed out, 0 means not
			bail_out = self.PROCESS(randok,g,S,alpha,X,Y)
			g = self.g
			alpha = self.alpha
			S = self.S
			lambdah = self.lambdah
			if bail_out == 0:
				###REPROCESS
				self.REPROCESS(g,S,alpha,X,Y)
				delta = self.delta
				b = self.b
				S = self.S
				g = self.g
				lambdah = self.lambdah
				alpha = self.alpha

			print(q)
	    
		print(5)

		while(self.delta > self.tau):
			self.REPROCESS(g,S,alpha,X,Y)
			delta = self.delta
			b = self.b
			S = self.S
			g = self.g
			lambdah = self.lambdah
			alpha = self.alpha
	      

		self.Xold = X
		self.alpha = alpha
		self.lambdah = lambdah
		self.g = g
		self.A = A
		self.B = B

		print(6)

		# print(len(S))




	# Implement this!
	def predict(self, X):
		self.X = X
		Xold = self.Xold

		#S = self.S
		alpha = self.alpha
		b = 0

		#print(len(X))

		Ypredvec = np.dot(alpha,np.dot(Xold,X.transpose()))

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

# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

val = np.loadtxt("val.csv", delimiter=',')
Xval = val[:, :2]
Yval = val[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0.344
N = 150
numsamples = 20000

# parameters for SMO
C = 1.15
tau = 1.50

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






# smomodel = SMO(numsamples,C,tau)
# starttime = time.time()
# smomodel.fit(X,Y)
# print("This took ",time.time() - starttime," seconds!")
# smomodel.visualize("smomodel.png", width=0, show_charts=True, save_fig=True, include_points=False)

# smomodel.predict(X)
# smomodel.accuracy(Y)

# smomodel.predict(Xval)
# smomodel.accuracy(Yval)

lasvmmodel = LASVM(numsamples,C,tau,initseednumber,iternumber)
starttime = time.time()
lasvmmodel.fit(X,Y)
print("This took ",time.time() - starttime," seconds!")
lasvmmodel.visualize("lasvmmodel.png", width=0, show_charts=True, save_fig=True, include_points=False)

lasvmmodel.predict(X)
lasvmmodel.accuracy(Y)

lasvmmodel.predict(Xval)
lasvmmodel.accuracy(Yval)






