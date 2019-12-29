import numpy as np
import random
import os.path
import matplotlib.pyplot as plt

class NNmodel:
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        self.W1 = .1 * np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = .1 * np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        self.noise = .3 * np.random.randn(input_size)
        self.batch_size = batch_size
        self.output_img = []
        self.filter = []
    
    def cross_entropy(self, y, t):
        # bs = self.batch_size
        # if y.shape[0] > 64:
        #     return -np.sum(t*np.log(y + 1e-10) + (1-t) * np.log((1-y)+1e-10)) /2 /y.shape[0]
        return -np.sum(t*np.log(y + 1e-10) + (1-t) * np.log((1-y)+1e-10)) /2 /y.shape[0]
    
    def relu(self, x):
        out = x.copy()
        mask = x < 0
        out[mask] = 0
        return out, mask
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        W1, W2 = self.W1, self.W2
        b1, b2 = self.b1, self.b2
        bs = self.batch_size
        # forward
        a1 = np.dot(x, W1) + b1
        h1, mask = self.relu(a1)
        a2 = np.dot(h1, W2) + b2
        y = self.sigmoid(a2)
        return y

    def show2D(self, x):
        W1 = self.W1
        b1 = self.b1
        a1 = np.dot(x, W1)
        h1, mask = self.relu(a1)
        output_2d = h1[:,:2]
        return output_2d


    def loss(self, x, t):
        y = self.predict(x)
        return self.cross_entropy(y, t)

    def drop(self, drop_x, ratio):
        drop_mask = np.array(random.sample(range(self.batch_size*128), \
                                            round(self.batch_size*128*ratio)))
        out = drop_x.flatten()
        out[drop_mask] = 0
        out = out.reshape(self.batch_size,128)
        return out, drop_mask


    def gradient(self, x, t, denoise=True, dropout=True):
        W1, W2 = self.W1, self.W2
        b1, b2 = self.b1, self.b2
        bs = self.batch_size

        # forward
        if denoise:
            x_with_noise = x + self.noise
            a1 = np.dot(x_with_noise, W1) + b1
        else:
            a1 = np.dot(x, W1) + b1

        h1, mask = self.relu(a1)
        
        if dropout:
            h1, drop_mask = self.drop(h1, 0.4)

        a2 = np.dot(h1, W2) + b2

        y = self.sigmoid(a2)

        # save result
        self.filter = W1.T
        self.output_img = y
        
        # backward
        dy = (y - t) / bs

        dW2 = np.dot(h1.T, dy)
        db2 = np.sum(dy, axis=0) / bs
        
        dh1 = np.dot(dy, W2.T)
        dh1[mask] = 0
        
        if dropout:
            out = dh1.flatten()
            out[drop_mask] = 0
            out = out.reshape(64,128)

        da1 = dh1

        dW1 = np.dot(x.T, da1)
        db1 = np.sum(da1, axis=0) / bs

        return dW1,db1,dW2,db2


def mnist_read():
    path = "mnist"
    mnist_file = ['train-images.idx3-ubyte','train-labels.idx1-ubyte'\
           ,'t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte']
    mnist_data = ['x_train','y_train','x_test','y_test']
    saved = ['x_train.npy','y_train.npy','x_test.npy','y_test.npy']

    for i in range(4):
        if os.path.exists(saved[i]):
            mnist_data[i] = np.load(saved[i])
        else:
            with open(path+"/"+mnist_file[i],'rb') as f:
                if (i%2 == 0):
                    mnist_data[i] = np.array([int(pix) for pix in f.read()[16:]]).reshape(-1, 784)
                else:
                    mnist_data[i] = np.array([int(pix) for pix in f.read()[8:]]).reshape(-1)
                np.save(saved[i], mnist_data[i])
    return mnist_data

def onehot(t):
    onehot_label = np.zeros((len(t),10))
    onehot_label[np.arange(len(t)),t] = 1
    return onehot_label

def plot2D(dataset, label, num_grid, size):
    # digit = [i for i,val in enumerate(label) if label[i] == 1]
    # x_2d = dataset[digit]
    x_2d = dataset[:size-1]
    output_2d = model.show2D(x_2d)
    x_w = (max(output_2d[:,0]) - min(output_2d[:,0]))
    x_h = (max(output_2d[:,1]) - min(output_2d[:,1]))
    rmi = (min(output_2d[:,0]))
    cmi = (min(output_2d[:,1]))
    gridc = (x_w/num_grid)
    gridr = (x_h/num_grid)
    g = max(gridc,gridr) * 0.5

    fig, ax = plt.subplots()
    tmp = []
    for i in range(num_grid):
        for j in range(num_grid):
            tmp = np.logical_and(np.logical_and((output_2d[:,0] > (rmi + gridc *i)), (output_2d[:,0] < (rmi+ gridc *(i+1)))), \
                                 np.logical_and((output_2d[:,1] > (cmi + gridr *j)), (output_2d[:,1] < (cmi + gridr *(j+1)))))
            if True in tmp:
                ax.scatter(output_2d[tmp][0,0], output_2d[tmp][0,1],facecolors='none',edgecolors='r',s= 10,zorder= 1)
                ax.imshow(x_2d[tmp][0].reshape(28,28), \
                    extent=(output_2d[tmp][0,0], output_2d[tmp][0,0]+g ,output_2d[tmp][0,1] ,output_2d[tmp][0,1]+g),zorder= 2,cmap='gray')
    ax.scatter(output_2d[:,0],output_2d[:,1],c='g',s= 0.3,zorder = -1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



if __name__ == '__main__':
    mnist_data = mnist_read()
    x_train, t_train, x_test, t_test = mnist_data[0]/255.0, onehot(mnist_data[1]),\
                                       mnist_data[2]/255.0, onehot(mnist_data[3])

    # hyper parameters 
    input_size = 784
    hidden_size = 128
    output_size = 784
    batch_size = 64
    lr = 0.01

    # setup NN model
    model = NNmodel(input_size, hidden_size, output_size, batch_size)

    iters = 2500
    iter_per_epoch = 500
    total = int(iters/iter_per_epoch)

    select = {1:True, 2:False}
    print("\n=================(b) & (c)======================")
    sel = int(input("Apply denoise and dropout?\nInput 1 for True, 2 for False : "))

    for i in range(iters+1):
        indices = np.random.choice(len(x_train), batch_size)
        x_batch = x_train[indices]
        t_batch = t_train[indices]
        dW1,db1,dW2,db2 = model.gradient(x_batch, x_batch, denoise=select[sel], dropout=select[sel])

        model.W1 -= lr * dW1
        model.b1 -= lr * db1
        model.W2 -= lr * dW2
        model.b2 -= lr * db2

        if i % iter_per_epoch == 0:
            train_loss = model.loss(x_batch, x_batch)
            test_loss = model.loss(x_test, x_test)
            print("Epoch[{}/{}]  |  label={}".format(int(i/iter_per_epoch),total, np.argmax(t_batch[:4],axis=1)))
            print("train loss, test loss  |  {:.4f}, {:.4f}".format(train_loss,test_loss))

            plt.figure()
            for j in range(4):
                plt.subplot(2,4,j+1)
                if j==0: plt.title('original')
                plt.imshow(x_batch[j].reshape(28,28),cmap='gray')
                plt.axis('off')
                
                plt.subplot(2,4,j+5)
                if j==0: plt.title('reconstructed')
                plt.imshow(model.output_img[j].reshape(28,28), cmap='gray')
                plt.axis('off')
            plt.show()

    print("The filters")
    plt.figure()
    for i in range(16):
        plt.subplot(4,4,i+1)
        if i==0: plt.title('filters')
        plt.imshow(model.filter[i].reshape(28,28), cmap='gray')
        plt.axis('off')

    plt.show()
    
    # plotting training set to 2D
    print("\n================(a)===================")
    print("Plotting training set on 2D plain")
    plot2D(x_train, mnist_data[1], 10,60000)
    print("Plotting testing set on 2D plain")
    plot2D(x_test, mnist_data[3], 10,10000)
