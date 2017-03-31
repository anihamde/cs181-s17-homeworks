# CS 181, Spring 2017
# Homework 4: Clustering
# Name: Anirudh Suresh
# Email: anirudh_suresh@college.harvard.edu

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rando

class KMeans(object):
	# K is the K in KMeans
	# useKMeansPP is a boolean. If True, you should initialize using KMeans++
	def __init__(self, K, useKMeansPP):
		self.K = K
		self.useKMeansPP = useKMeansPP

	# X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
	def fit(self, X):
		randsampvals = rando.sample(range(len(X)),self.K)
		centroidmeans = X[0:self.K]

		for a in range(0,self.K):
			centroidmeans[a] = X[randsampvals[a]]

		clustermat = np.repeat(0,len(X))
		validator = centroidmeans*3

		countervar = 0

		# self.clustermat = clustermat
		# self.centroidmeans = centroidmeans
		# self.X = X

		while(not np.array_equal(validator,centroidmeans)):#np.all(validator != centroidmeans):

			print(countervar)
			# print(self.get_error())

			validator = centroidmeans

			for j in range(0,len(X)):
				normvals = np.repeat(0,self.K)
				for k in range(0,self.K):
					normvals[k] = np.linalg.norm(X[j] - centroidmeans[k])
				clustermat[j] = np.argmin(normvals)

				# Xarray = np.reshape(X,(28*28))
				# for i in range(0,len(Xarray)):
				# 	normvals = np.repeat(0,self.K)
				# 	for k in range(0,self.K):
				# 		normvals[k] = np.linalg.norm((np.repeat(Xarray[i],self.K) - centroidmeans)[k])
				# 	clustermat[j,i] = np.argmin(normvals)

			centroidmeans = X*0
			count = np.repeat(0,self.K)
			for j in range(0,len(X)):
				centroidmeans[clustermat[j]] += X[j]
				count[clustermat[j]] += 1

			for s in range(0,self.K):
				centroidmeans[s] = centroidmeans[s]/count[s]

			countervar += 1

			# self.clustermat = clustermat
			# self.centroidmeans = centroidmeans

		self.clustermat = clustermat
		self.centroidmeans = centroidmeans
		self.X = X

		self.get_error()

	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_mean_images(self):
		for i in range(0,self.K):
			print("Mean of class ",i,"!")
			self.create_image_from_array(self.centroidmeans[i])
		return

	# This should return the arrays for D images from each cluster that are representative of the clusters.
	def get_representative_images(self, D):
		for i in range(0,self.K):
			indexlist = []
			scores = []
			imagenumb = 0
			for j in range(0,len(self.X)):
				if(self.clustermat[j] == i):
					indexlist.append(j)

			for k in range(0,len(indexlist)):
				scores.append(np.linalg.norm(self.X[indexlist[k]]-self.centroidmeans[i]))
			
			while(imagenumb < D):
				valuableindex = np.argmin(scores)

				print(imagenumb+1, "th representative image out of ",D," of class ",i,"!")

				self.create_image_from_array(self.X[indexlist[valuableindex]])

				# print(len(indexlist))

				indexlist = np.delete(indexlist,valuableindex)
				# print(len(indexlist))
				# print(len(scores))

				scores = np.delete(scores,valuableindex)
				# print(len(scores))

				imagenumb += 1

	def get_error(self):
		sumsq = 0
		for i in range(0,len(self.X)):
			sumsq += (np.linalg.norm(self.X[i]-self.centroidmeans[self.clustermat[i]]))**2
		print("The average sum of squares error is ",sumsq/len(self.X),"!")
		return


	# img_array should be a 2D (square) numpy array.
	# Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
	# However, we do ask that any images in your writeup be grayscale images, just as in this example.
	def create_image_from_array(self, img_array):
		plt.figure()
		plt.imshow(img_array, cmap='Greys_r')
		plt.show()
		return

# This line loads the images for you. Don't change it! 
pics = np.load("images.npy", allow_pickle=False)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.
K = 10
KMeansClassifier = KMeans(K=10, useKMeansPP=False)
KMeansClassifier.fit(pics)
# KMeansClassifier.create_image_from_array(pics[1])

KMeansClassifier.get_mean_images()
KMeansClassifier.get_representative_images(2)





