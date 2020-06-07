import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GMM_EM():
    def __init__(self, k):
        self.k = k
        self.pi = np.random.randn(k)
        self.pi /= np.sum(self.pi)
        self.li = []
        
    def train(self, data, epoch, plot = True):
        self.n, self.feature = data.shape
        if self.feature != 2 and plot:
            print('Plot function only suitable for 2d data')
            plot = False
        self.mu = np.random.randn(self.k, self.feature)
        sig = np.eye(self.feature)
        sig = [np.array([sig])]*self.k
        self.sig = np.concatenate(sig)
        figure = plt.figure()
        for i in range(epoch):
            self.e_step(data)
            self.N = np.sum(self.r, axis = 0)
            self.m_step(data)
            if plot:
                self.plot_contour(data, figure, i)
        if plot:
            plt.show()
            
    def e_step(self, data):
        r = self.likelihood(data)
        self.r = r/np.sum(r*self.pi, axis = 1, keepdims=True)

    def m_step(self, data):
        mu = np.zeros((self.k, self.feature))
        sig = np.zeros((self.k, self.feature, self.feature))
        for i in range(self.k):
            mu[i, :] = 1/self.N[i]*np.sum(self.r[:,i:i+1]*data, axis = 0) #update mu
            diff = data - mu[i, :]
            sig[i, :, :] = 1/self.N[i]*np.matmul(diff.transpose(), self.r[:,i:i+1]*diff) #update sigma
        self.mu = mu
        self.sig = sig
        self.pi = self.N/self.n
        
    def likelihood(self, data):
        r = np.zeros((self.n, self.k))
        pro_l = np.zeros((self.n, self.k))
        for i in range(self.k):
            pro = multivariate_normal.pdf(data, self.mu[i], self.sig[i])
            pro_l[:, i] = pro.transpose()
            r[:,i] = self.pi[i]*pro
        l = np.sum(pro_l, axis = 1)
        self.li.append(np.sum(-np.log(l)))
        return r
        
    def plot_li(self):
        plt.plot(range(len(self.li)), self.li)
        plt.show()
        
    def plot_contour(self, data, figure, epoch):
        # define grid.
        figure.clear()
        for i in range(self.k):
            x = np.linspace(self.mu[i,0]-3, self.mu[i,0]+3, 100)
            y = np.linspace(self.mu[i,1]-3, self.mu[i,1]+3, 100)
            ## grid the data.
            X, Y = np.meshgrid(x, y)
            Z = np.dstack((X, Y))
            Z = Z.reshape(-1, 2)
            Z = multivariate_normal(self.mu[i], self.sig[i]).pdf(Z)
            Z = Z.reshape(100,100)
            plt.contour(X,Y,Z)
        plt.scatter(data[:,0], data[:,1])
        plt.colorbar() # draw colorbar
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.title('Epoch'+str(epoch+1))
        plt.pause(0.001)
        
def main():
    data1 = np.random.normal(0, 0.5, (50,2))
    data2 = np.random.normal(3, 0.9, (50,2))
    data3 = np.random.normal(-3, 0.2, (50,2))
    data4 = np.random.normal(-1, 0.2, (50,2))
    data5 = np.random.normal(1, 0.2, (50,2))
    data = np.concatenate((data1,data2,data3,data4,data5))
    em = GMM_EM(5)
    em.train(data, 50, True)

if __name__ == '__main__':
    main()

