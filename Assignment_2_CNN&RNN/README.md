## Steps to run the code
### To directly run the code under the "Part 1", "Part 2" and the "Part 3" directory, set those directories as the Source Root.
### 1. Part 1
#### Task1 & Task2
(1) Feel free to run the Part1.ipynb <br>
(2) To train the model on the default make_moons dataset, you can 
also run the train method inside the main function of pytorch_train_mlp.py
```
def main():
    """
    Main function
    """
    acc_test,x_axis = train('sgd')
    # acc_test,x_axis = train('batch')
    print(acc_test)
```
(3) To run the model on different dataset, change the dataset like this:
```
from sklearn.datasets import make_circles
X, Y = make_circles(1000, random_state=25)
x_trainc, x_testc, y_trainc, y_testc = train_test_split(X, Y, test_size=0.2, random_state=42)
y_train_one_hotc = np.eye(2)[y_trainc]
y_test_one_hotc = np.eye(2)[y_testc]
```
#### Task3
The model is defined in model_CIFAR10.py while the training method
is defined train_model_CIFAR10.py
(1) Get CIFAR10 dataset, please change the download to True if
you haven't download the CIFAR10 dataset to the data directory
```
def get_cifar10(batch_size):
```
(2) Train the model
```
net = mc.MC()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net,train_x_axis,train_loss_lst,x_axis_test,test_loss_lst=tmc.train(model=net,train_set=train_loader,criterion=criterion,optimizer=optimizer,max_epoch=15,validate_batch=2000,test_set=test_loader,batch_size=5)
test_perc=tmc.test(model=net,test_set=test_loader,classes=classes,batch_size=5)
```
(3) To customize the training
```
model = mc.MC()
                model_trained = tmc.train_tune(model=model,train_loader=train_loader,test_loader=test_loader,criterion=criterion,optimizer=opt,lr=lr,max_epoch=epoch,batch_size=bs)
                percentage,accuracy = tmc.test(model_trained,test_loader,classes,batch_size=bs)
```
### 2. Part 2
#### Task1
The model is defined in cnn_model.py while the training method
is defined in cnn_train.py
#### Task2
(1) Feel free to run the Part2.ipynb
(2) The train method in cnn_train.py is for training the model
```
cnn = CNN(32,10)
cnn, x_axis_train, loss_lst_train, x_axis_test, loss_lst_test,accu_lst_test=cnn_train.train(net=cnn,max_epoch=3)
```
It contains the testing phrase, so the test loss and accuracy is returned together with
training accuracy.
### 3. Part 3
#### Task1
The model is defined in vanilla_rnn.py while the training method is in train.py.
#### Task2
Since the predefined coding structure uses argparse, the running of the training is as
follows:
```
parser = argparse.ArgumentParser()

# Model params
parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence') # 10
parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')# 10
parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps') #10000
parser.add_argument('--max_norm', type=float, default=10.0)

config = parser.parse_args(args=[])
x_axis,loss_lst,accu_lst=train.train(config)
```
### 4. All the ipynb includes functions to show the accuracy figure