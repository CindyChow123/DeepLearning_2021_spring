# Instructions to run the code
## Put all the code inside one Source Root directory in a project
## Part 1 (Please import perceptron.py)
### 1. To generate the data and train the perceptron model, run the following code inside Report.ipynb
```
mean=[(1,2),(1,1)]
cova=[[[1,0],[0,1]],[[1,0],[0,1]]]
X_train,X_test,y_train,y_test = distribution_sampling(mean:{list:2},cova{list:2})
```
### 2. To train and print out the classification accuracy on the test set, run the following code inside Report.ipynb
```
perceptron_training(X_train:ndarray,y_train:ndarray,X_test:ndarray,y_test:ndarray)
```
### 3. To examine the perceptron's performance under different means and variance condition, define a code block with the above two code part together

## Part 2 (Please import train_mlp_numpy.py)
### 1. Predefined data has been generated inside method 'train' in train_mlp_numpy.py
### 2. To train the MLP, use the following method, where x denotes the distance of x-label, test_acc and train_acc denotes test, train accuracy data point respectively:
```
# input 'batch' or 'sgd' to choose between batch gradient descent or stochastic gradient descent.
x, test_acc, train_acc = tmn.main('batch')
```
### 3. To draw the accuracy curve, use matplotlib.pyplot like this
```
import train_mlp_numpy as tmn
import matplotlib.pyplot as plt
# %matplotlib inline
x, test_acc, train_acc = tmn.main('sgd')
plt.plot(x,test_acc,'r--',label='test')
plt.plot(x,train_acc,'g--',label='train')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```

