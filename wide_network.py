import numpy as np
import os.path
import matplotlib.pyplot as plt

class NNmodel:
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        self.W1 = .1 * np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = .1 * np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        self.batch_size = batch_size
    
    def cross_entropy(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        if t.size == y.size:
            t = t.argmax(axis = 1)
        bs = self.batch_size
        # return -np.sum(t*np.log(y + 1e-10) + (1-t) * np.log((1-y)+1e-10)) /2 /bs
        return -np.sum(np.log(y[np.arange(bs), t] + 1e-10)) / bs
        # return -np.sum(t*np.log(y + 1e-10)) / bs

    def relu(self, x):
        mask = x<0
        out = x.copy()
        out[mask] = 0
        return out, mask

    def relu_grad(self, x, mask):
        out = x.copy()
        out[mask] = 0
        out[x>=0] = 1
        return out

    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        x = x-np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def predict(self, x):
        W1, W2 = self.W1, self.W2
        b1, b2 = self.b1, self.b2
        
        # forward
        a1 = np.dot(x, W1) + b1
        h1,mask = self.relu(a1)
        a2 = np.dot(h1, W2) + b2
        y = self.softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return self.cross_entropy(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        W1, W2 = self.W1, self.W2
        b1, b2 = self.b1, self.b2
        bs = self.batch_size

        # forward
        a1 = np.dot(x, W1) + b1
        h1, mask = self.relu(a1)
        a2 = np.dot(h1, W2) + b2
        y = self.softmax(a2)
        
        # backward
        dy = (y - t) / bs
        dW2 = np.dot(h1.T, dy)
        db2 = np.sum(dy, axis=0)
        
        dh1 = np.dot(dy, W2.T)
        da1 = self.relu_grad(a1,mask) * dh1

        dW1 = np.dot(x.T, da1)
        db1 = np.sum(da1, axis=0)

        return dW1, db1, dW2, db2


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


if __name__ == '__main__':
    mnist_data = mnist_read()
    x_train, t_train, x_test, t_test = mnist_data[0]/255.0, onehot(mnist_data[1]),\
                                       mnist_data[2]/255.0, onehot(mnist_data[3])
    # hyper parameters 
    input_size = 784
    hidden_size = 256
    output_size = 10
    batch_size = 64
    lr = 0.1

    # setup NN model
    model = NNmodel(input_size, hidden_size, output_size, batch_size)

    iters = 10000
    iter_per_epoch = 1000
    total = int(iters/iter_per_epoch)
    
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    print("\nWide hidden layer network")
    print("input layer:  {}".format(input_size))
    print("hidden layer: {}".format(hidden_size))
    print("output layer: {}".format(output_size))
    print("batch size:   {}".format(batch_size))
    print("Training 10000 iterations, 1000 per epoch.\n")
    for i in range(iters+1):
        indices = np.random.choice(len(x_train), batch_size)
        indices_t = np.random.choice(len(x_test), batch_size)
        x_batch = x_train[indices]
        t_batch = t_train[indices]
        x_test_batch = x_test[indices_t]
        t_test_batch = t_test[indices_t]
    
        dW1, db1, dW2, db2 = model.gradient(x_batch, t_batch)
        model.W1 -= lr * dW1
        model.b1 -= lr * db1
        model.W2 -= lr * dW2
        model.b2 -= lr * db2

        if i % iter_per_epoch == 0:
            train_loss = model.loss(x_batch, t_batch)
            train_loss_list.append(train_loss)
            test_loss = model.loss(x_test_batch, t_test_batch)
            test_loss_list.append(test_loss)

            train_acc = model.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            test_acc = model.accuracy(x_test, t_test)
            test_acc_list.append(test_acc)
            print("Epoch[{}/{}] train acc ,test acc | {:.4f} ,{:.4f}  /  train loss, test loss | {:.2f} .{:.2f}"\
                    .format(int(i/iter_per_epoch),total,train_acc, test_acc, train_loss, test_loss))        


    # Plot acc
    plt.figure()
    plt.subplot(1,2,1)
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc',linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.xlim(0, total)
    plt.xticks(range(len(train_acc_list)))
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')

    # Plot loss
    plt.subplot(1,2,2)
    x = np.arange(len(train_loss_list))
    plt.plot(x, train_loss_list, label='train loss')
    plt.plot(x, test_loss_list, label='test loss',linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xlim(0, total)
    plt.xticks(range(len(train_loss_list)))
    plt.legend(loc='upper right')
    plt.show()

    

