# Fully Connected Neural Network
---
#### development environment：
```c
Linux version 4.15.0-36-generic (buildd@lgw01-amd64-031)
Description:	      Ubuntu 18.04.1 LTS
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz x 12
GPU :                GeForce GTX 1080 Ti
memory:              32G
L1d cache:           32K
L1i cache:           32K
L2 cache:            256K
L3 cache:            12288K
Python version:      Python 3.6.5
```

### 1.
### (a) wide hidden layer

- Hyperparameters
```python
input layer  neurons: 784
hidden layer neurons: 256
output layer neurons: 10
learning rate: 0.1
batch size : 64
```
- training, testing accuracy
```
Epoch[0/10]  train acc ,test acc | 0.1541 ,0.1505 
Epoch[1/10]  train acc ,test acc | 0.9403 ,0.9388 
Epoch[2/10]  train acc ,test acc | 0.9588 ,0.9533 
Epoch[3/10]  train acc ,test acc | 0.9669 ,0.9612 
Epoch[4/10]  train acc ,test acc | 0.9731 ,0.9667 
Epoch[5/10]  train acc ,test acc | 0.9782 ,0.9697 
Epoch[6/10]  train acc ,test acc | 0.9816 ,0.9712 
Epoch[7/10]  train acc ,test acc | 0.9842 ,0.9731 
Epoch[8/10]  train acc ,test acc | 0.9861 ,0.9739 
Epoch[9/10]  train acc ,test acc | 0.9872 ,0.9742 
Epoch[10/10] train acc ,test acc | 0.9898 ,0.9748
```

- accuracy and loss curve
![](https://i.imgur.com/nQtPtya.png)



### (b) deep hidden layer

- Hyperparameters
```python
input layer  neurons: 784
hidden layer1 neurons: 204
hidden layer2 neurons: 202
output layer neurons: 10
learning rate: 0.1
batch size : 64
```

- training, testing accuracy
```
Epoch[0/10]  train acc ,test acc | 0.1363 ,0.1348 
Epoch[1/10]  train acc ,test acc | 0.9540 ,0.9499 
Epoch[2/10]  train acc ,test acc | 0.9688 ,0.9629 
Epoch[3/10]  train acc ,test acc | 0.9733 ,0.9648 
Epoch[4/10]  train acc ,test acc | 0.9818 ,0.9702 
Epoch[5/10]  train acc ,test acc | 0.9867 ,0.9750 
Epoch[6/10]  train acc ,test acc | 0.9891 ,0.9762 
Epoch[7/10]  train acc ,test acc | 0.9906 ,0.9755 
Epoch[8/10]  train acc ,test acc | 0.9919 ,0.9758 
Epoch[9/10]  train acc ,test acc | 0.9953 ,0.9781 
Epoch[10/10] train acc ,test acc | 0.9951 ,0.9782
```

- accuracy and loss curve
![](https://i.imgur.com/mzjLRzD.png)



### 2. Implement an autoencoder (AE) to learn the representation of the MNIST datasets.

### (a) Show the results of the AE-based dimension reduction such as HW3-A.
### training set
- all digit of top 10000 in training set
![](https://i.imgur.com/irmSsiu.png)
- digit 0
![](https://i.imgur.com/P9x0S0D.png)

- digit 1
![](https://i.imgur.com/BW714lq.png)


### testing set
- all
![](https://i.imgur.com/SCjEzvm.png)

- digit 0
![](https://i.imgur.com/oJVSl83.png)

- digit 1
![](https://i.imgur.com/c7PmcVt.png)

```














```

### (b) Visualize the reconstruction results and the filters. 
- Hyperparameters
```python
input layer  neurons: 784
hidden layer neurons: 128
output layer neurons: 784
learning rate: 0.01
batch size : 64
```

```
Epoch[0/5]  |  label=[7 1 4 2]
train loss, test loss  |  266.5626, 267.8676
Epoch[1/5]  |  label=[7 3 2 9]
train loss, test loss  |  48.1467, 49.5476
Epoch[2/5]  |  label=[5 8 8 1]
train loss, test loss  |  40.5914, 41.3452
Epoch[3/5]  |  label=[0 6 0 4]
train loss, test loss  |  36.6603, 37.0315
Epoch[4/5]  |  label=[3 9 1 9]
train loss, test loss  |  34.2981, 34.9079
Epoch[5/5]  |  label=[2 2 5 9]
train loss, test loss  |  33.2759, 33.7161
```


- Epoch[0/5]
```
label=[1 9 4 2]
train loss, test loss  |  260.8990, 262.6952
```
![](https://i.imgur.com/mdT6NHR.png)

- Epoch[1/5]
```
label=[1 2 8 0]
train loss, test loss  |  35.8100, 36.0440
```
![](https://i.imgur.com/LeHzOIV.png)

- Epoch[2/5]
```
label=[7 1 2 5]
train loss, test loss  |  32.1337, 31.8156
```
![](https://i.imgur.com/vrUUKLt.png)

- Epoch[3/5]
```
label=[7 3 0 8]
train loss, test loss  |  31.7762, 30.6087
```
![](https://i.imgur.com/4zMAEuh.png)

- Epoch[4/5]
```
label=[3 8 4 3]
train loss, test loss  |  29.6784, 29.4307
```
![](https://i.imgur.com/BdzLTNZ.png)

- Epoch[5/5]
```
label=[8 9 9 5]
train loss, test loss  |  27.9382, 28.8068
```
![](https://i.imgur.com/R3iOe2r.png)


- Filter
![](https://i.imgur.com/E29MvRB.png)



### （c）Apply denoise and dropout mechanism, and visualize the reconstruction results and the filters. (10%, Bonus)

```
(env) forest@server:~/env/mlhw4$ py autoencoder.py 
Apply denoise and dropout?
Input 1 for True, 2 for False : 1
Epoch[0/5]  |  label=[9 6 9 1]
train loss, test loss  |  274.4898, 275.0325
Epoch[1/5]  |  label=[2 8 1 5]
train loss, test loss  |  58.2506, 66.8891
Epoch[2/5]  |  label=[8 5 4 3]
train loss, test loss  |  60.0102, 64.2880
Epoch[3/5]  |  label=[2 0 3 6]
train loss, test loss  |  64.3706, 68.0223
Epoch[4/5]  |  label=[2 6 6 0]
train loss, test loss  |  66.2162, 69.2971
Epoch[5/5]  |  label=[4 1 4 2]
train loss, test loss  |  54.8299, 60.2125
```

- Epoch[5/5] 
```
label=[2 5 0 0]
train loss, test loss  |  61.5583, 60.5595
```
![](https://i.imgur.com/G5jW9z0.png)


- Filter with denoise and dropout
![](https://i.imgur.com/rdQU9gI.png)
