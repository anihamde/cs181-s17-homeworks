import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
    
    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def LogLoss(self, X, Y, w):
        logloss = 0
        for i in range(0,59):
            numerator = np.exp(np.dot(w[Y[i]],X[i]))
            denominator = 0
            for n in range(0,3):
                denominator += np.exp(np.dot(w[n],X[n]))
            logloss += -(np.log(numerator/denominator))
        print(logloss)
        return logloss

    def checkclass(self,y,k,i):
        if y[i]==k:
            return 1
        else:
            return 0

    # TODO: Implement this method!
    def fit(self, X, C):
        qnumb = []
        qloss = []
        self.X = X
        self.C = C
        w = (1/9) *np.arange(9).reshape(3,3)
        eta = self.eta
        lambda_parameter = self.lambda_parameter

        X = np.append(X,np.ones([len(X),1]), 1)
        V = w.T
        iter = 10000
        while (iter > 0):
            for k in range(0,3):
                correc = 0
                for i in range(0,59):
                    denominator = 0
                    for l in range(0,3):
                        denominator += np.exp(np.dot((V[l]).T, X[i]))
                    correc += (np.exp(np.dot((V[k]).T, X[i]))/denominator - self.checkclass(C,k,i)) * X[i] + self.lambda_parameter * w[k] * X[i]
                w[k] = (V[k] - eta * correc).T
            iter -= 1
            if (iter+1) % 100 == 0:
                qnumb.append(10000-iter+1)
                self.LogLoss(X, C, w)
                qloss.append(self.LogLoss(X, C, w))

        plt.scatter(qnumb[:], qloss[:])
        plt.savefig("4plot.png")



        self.w = w

        return


        # eta = self.eta
        # lambda_parameter = self.lambda_parameter
        # X = np.append(X,np.ones([len(X),1]),1)

        # self.X = X
        # self.C = C

        # w = (1/9) *np.arange(9).reshape(3,3)

        # #rowd = [0,0,0]#np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])

        # iter = 0
        # while(iter < 100):
        #     for j in range(0, 3):
        #         rowd = [0,0,0]
        #         for i in range(0, 59):
        #             denominator = 0

        #             for k in range(0, 3):
        #                 denominator += np.exp(np.dot((w[k]).T,X[i]))

        #             rowd += ((np.exp(np.dot((w[j]).T,X[i]))/denominator)-self.checkclass(C,i,j))*X[i] + self.lambda_parameter*w[j]*X[i]

        #     # for k in range(0, 3):
        #     #     for i in range(0, 59):
        #     #         rowd[k] += ((np.exp(np.dot((w[k]).T,X[i]))/denominator) - self.checkclass(C,i,k))*X[i] + self.lambda_parameter * w[k] * X[i]

        #         w[j] = (w[j] - eta * rowd)
        #     # print(denominator)
        #     iter += 1
       
        # self.w = w
        # ####print(w)
        # # return np.array(b)

        return

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        X_to_predict = np.append(X_to_predict,np.ones([len(X_to_predict),1]),1)

        #print(X_to_predict)
        w = self.w
        b = []
        Y = np.dot(X_to_predict,w)

        # denominator = 0


        # for j in range(0, 3):
        #     for i in range(0,X_to_predict.shape[0]):
        #         denominator = 0
        #         for k in range(0, 3):
        #             denominator += np.exp(np.dot((w[k]).T,X_to_predict[i]))
                
        #         Y[i,j] = np.exp(np.dot((w[j]).T,X_to_predict[i]))/denominator


        # for i in range(0,Y.shape[0]):
        #     for k in range(0,3):
        #         Y[i,k] = np.exp(np.dot((w[k]).T,X_to_predict[i]))/denominator

        for i in range(0,Y.shape[0]):
            val=np.argmax(Y[i])
            # val = 0
            # if Y[i,0] > Y[i,1] and Y[i,0] > Y[i,2]:
            #     val = 0
            # elif Y[i,1] > Y[i,0] and Y[i,1] > Y[i,2]:
            #     val = 1
            # elif Y[i,2] > Y[i,0] and Y[i,2] > Y[i,1]:
            #     val = 2
            b.append(val)

        #print(np.array(b).shape[0])
        # print(b)
        ####print(b)

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
        # print(np.array(Y).shape)
        # return np.array(Y)

    def visualize(self, output_file, width=2, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

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
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
