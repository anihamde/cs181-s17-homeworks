from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def checkclass(self,y,k,i):
        if y[i]==k:
            return 1
        else:
            return 0

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        ysum0 = [0,0]
        yxsum0 = 0

        ysum1 = [0,0]
        yxsum1 = 0

        ysum2 = [0,0]
        yxsum2 = 0

        for i in range(0,X.shape[0]):
            ysum0 += self.checkclass(Y,0,i)*X[i]
            yxsum0 += self.checkclass(Y,0,i)

        mu0 = ysum0/yxsum0

        for i in range(0,X.shape[0]):
            ysum1 += self.checkclass(Y,1,i)*X[i]
            yxsum1 += self.checkclass(Y,1,i)

        mu1 = ysum1/yxsum1

        for i in range(0,X.shape[0]):
            ysum2 += self.checkclass(Y,2,i)*X[i]
            yxsum2 += self.checkclass(Y,2,i)

        mu2 = ysum2/yxsum2



        covMLE = 0

        covMLE0 = 0
        covMLE1 = 0
        covMLE2 = 0

        for i in range(0,X.shape[0]):
            covMLE0 += (1/X.shape[0])*np.matmul(np.matrix((X[i] - mu0)).T,np.matrix((X[i]-mu0)))*self.checkclass(Y,0,i)
        for i in range(0,X.shape[0]):
            covMLE1 += (1/X.shape[0])*np.matmul(np.matrix((X[i] - mu1)).T,np.matrix((X[i]-mu1)))*self.checkclass(Y,1,i)
        for i in range(0,X.shape[0]):
            covMLE2 += (1/X.shape[0])*np.matmul(np.matrix((X[i] - mu2)).T,np.matrix((X[i]-mu2)))*self.checkclass(Y,2,i)

        covMLE = covMLE0+covMLE1+covMLE2
        # print(mu0)
        # print(mu1)
        # print(mu2)
        #print(covMLE)

        # print((X[1] - mu0))
        # print((X[1]-mu0).T)
        # print(np.dot((X[1] - mu0),(X[1]-mu0).T))
        # print(np.dot((X[1] - mu0).T,(X[1]-mu0)))

        self.mu0 = mu0
        self.mu1 = mu1
        self.mu2 = mu2
        self.covMLE = covMLE
        self.covMLE0 = covMLE0
        self.covMLE1 = covMLE1
        self.covMLE2 = covMLE2

        if(self.isSharedCovariance):
            covMLE0 = covMLE
            covMLE1 = covMLE
            covMLE2 = covMLE

        countclass = [0, 0, 0]
        for j in range(0,Y.shape[0]):
            countclass[Y[j]] += 1

        je = 1
        for i in range(0,X.shape[0]):
            power = self.checkclass(Y,0,i)
            je *= ((countclass[0]/Y.shape[0])*multivariate_normal.pdf(X[i],mu0,covMLE0))**(power)
            power = self.checkclass(Y,1,i)
            je *= ((countclass[1]/Y.shape[0])*multivariate_normal.pdf(X[i],mu1,covMLE1))**(power)
            power = self.checkclass(Y,2,i)
            je *= ((countclass[2]/Y.shape[0])*multivariate_normal.pdf(X[i],mu2,covMLE2))**(power)
        print("LIKELIHOOD:", np.log(je))

        return

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        mu0 = self.mu0
        mu1 = self.mu1
        mu2 = self.mu2
        covMLE = self.covMLE
        covMLE0 = self.covMLE0
        covMLE1 = self.covMLE1
        covMLE2 = self.covMLE2
        Y = self.Y

        if(self.isSharedCovariance):
            covMLE0 = covMLE
            covMLE1 = covMLE
            covMLE2 = covMLE

        b = []
        for i in range(0,X_to_predict.shape[0]):
            b.append(np.argmax([multivariate_normal.pdf(X_to_predict[i],mu0,covMLE0),multivariate_normal.pdf(X_to_predict[i],mu1,covMLE1),multivariate_normal.pdf(X_to_predict[i],mu2,covMLE2)]))
            if i%10000 == 0:
                print(i)    

        return np.array(b)
        # # The code in this method should be removed and replaced! We included it just so that the distribution code
        # # is runnable and produces a (currently meaningless) visualization.
        # Y = []
        # for x in X_to_predict:
        #     val = 0
        #     if x[1] > 4:
        #         val += 1
        #     if x[1] > 6:
        #         val += 1
        #     Y.append(val)
        # return np.array(Y)

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .005), np.arange(y_min,
            y_max, .005))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))

        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
