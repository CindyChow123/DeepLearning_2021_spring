# import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from torchvision import datasets, transforms,utils
import model_CIFAR10
from tqdm import tqdm

# Total train size(50000)/batch size(5)=10000 mini batches
# validate on each 2000 mini batches
MAX_EPOCH=2
BATCH_SIZE=5
MINI_BATCH=2000
TEST_SIZE=10000

def imgshow(imgs):
    # imshow need (h,w,channel), so patch together and transpose dimension
    npimgs = utils.make_grid(imgs).numpy()
    plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.show()

def train_tune(model,train_loader,test_loader,criterion,optimizer,lr,max_epoch,validate_batch=0,batch_size=4):
    # ['SGD', 'RMSprop', 'Adagrad', 'Adam']
    if optimizer=='sgd':
        optimizer=optim.SGD(model.parameters(),lr=l)
    elif optimizer=='RMSprop':
        optimizer=optim.RMSprop(model.parameters(),lr=lr)
    elif optimizer=='Adagrad':
        optimizer=optim.Adagrad(model.parameters(),lr=lr)
    elif optimizer=='Adam':
        optimizer=optim.Adam(model.parameters(),lr=lr)
    else:
        optimizer=optim.SGD(model.parameters(),lr=lr)
    model,x_axis,loss_lst,x_axis_test,loss_lst_test=train(model,train_loader,criterion,optimizer,max_epoch,validate_batch,test_loader,batch_size=batch_size)
    return model


def train(model,train_set,criterion,optimizer,max_epoch,validate_batch,test_set,batch_size):
    avg_loss=0
    x_axis=[]
    x_axis_test=[]
    loss_lst=[]
    loss_lst_test=[]
    with tqdm(total=50000*max_epoch) as pbar:
        pbar.set_description('model training:')
        for epoch in range(max_epoch):
            for index, data in enumerate(train_set,0):
                imgs, labels = data

                optimizer.zero_grad()

                results = model(imgs)
                loss=criterion(results,labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                if validate_batch!=0 and index % validate_batch == validate_batch-1:  # print every 2000 mini-batches
                    # print('[epoch %d, total %5d] loss: %.3f' %
                    #       (epoch + 1, (index+1)*BATCH_SIZE+epoch*50000, avg_loss / 2000))
                    # training loss
                    x_axis.append((index+1)*batch_size+epoch*50000)
                    loss_lst.append(avg_loss/validate_batch)
                    avg_loss = 0.0
                    pbar.update(validate_batch*batch_size)
                    # test loss
                else:
                    pbar.update(batch_size)
            if validate_batch != 0:
                with torch.no_grad():
                    test_avg_loss = 0
                    for data in test_set:
                        images, labels = data
                        outputs = model(images)
                        test_loss = criterion(outputs, labels)
                        test_avg_loss += test_loss.item()
                    loss_lst_test.append(test_avg_loss / 2000) # TEST SIZE/BATCH=sum of how many loss
                    x_axis_test.append(50000*(epoch+1))


    print("Training finish\n")
    return model,x_axis,loss_lst,x_axis_test,loss_lst_test

def test(model,test_set,classes,batch_size):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    percentage=[]
    with torch.no_grad():
        for data in test_set:
            images, labels = data
            outputs = model(images)
            _, class_label = torch.max(outputs.data, 1)
            total += labels.size(0)
            # tensor [true,false]; tensor[number]
            is_true = (class_label==labels).data
            correct += is_true.sum().item()
            for i in range(batch_size):
                label=class_label[i]
                class_correct[label]+=is_true[i]
                class_total[label]+=1
    for i in range(10):
        if class_total[i]!=0:
            # print('Accuracy of %5s : %2d %%' % (
                # classes[i], 100 * class_correct[i] / class_total[i]))
            percentage.append(class_correct[i]/class_total[i])
        else:
            # print('Accuracy of %5s : %2d %%' % (
            #     classes[i], 0))
            percentage.append(0)
    accuracy=100*correct/total
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    return percentage,accuracy

def get_cifar10(batch_size):
    # transform PIL image to tensors(normalize not support pil), normalize - dataloader needs
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # load train and test img set
    train_set = datasets.CIFAR10(root='../data', train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform)
    # iterate
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    # classes
    classes = ('airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader,test_loader,classes


# if __name__ == '__main__':
#     # transform PIL image to tensors(normalize not support pil), normalize - dataloader needs
#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     # load train and test img set
#     train_set = datasets.CIFAR10(root='../data', train=True, download=False,transform=transform)
#     test_set = datasets.CIFAR10(root='../data', train=False, download=False,transform=transform)
#     # iterate
#     train_loader = DataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
#     test_loader = DataLoader(dataset=test_set,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
#     # classes
#     classes = ('airplane', 'automobile', 'bird', 'cat',
#                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     # data_iter = iter(train_loader)
#     # imgs,labels=data_iter.next()
#
#     # model definition
#     net = model_CIFAR10.MC()
#     criterion = nn.CrossEntropyLoss()
#     # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#     optimizer = optim.Adagrad(net.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0,
#                               eps=1e-10)
#     net,train_x_axis,train_loss_lst,test_x_axis,test_loss_lst=train(model=net,train_set=train_loader,criterion=criterion,optimizer=optimizer,max_epoch=MAX_EPOCH,validate_batch=MINI_BATCH,test_set=test_loader)
#
#     test_perc=test(model=net,test_set=test_loader,classes=classes)

if __name__=="__main__":
    batch_size = [4]
    max_epoch_lst = [5]
    optimizer_lst = ['sgd']
    lr_lst = [0.001]
    criterion = nn.CrossEntropyLoss()
    accu_best = 0
    model_best = None
    param_best = {}
    for bs in batch_size:
        train_loader, test_loader, classes = get_cifar10(batch_size=bs)
        for epoch in max_epoch_lst:
            for opt in optimizer_lst:
                for lr in lr_lst:
                    model = model_CIFAR10.MC()
                    model_trained = train_tune(model=model, train_loader=train_loader, test_loader=test_loader,
                                                   criterion=criterion, optimizer=opt, lr=lr, max_epoch=epoch,batch_size=bs)
                    percentage, accuracy = test(model_trained, test_loader, classes,batch_size=bs)
                    if accuracy > accu_best:
                        accu_best = accuracy
                        model_best = model_trained
                        param_best = {'bs': bs, 'epoch': epoch, 'opt': opt, 'lr': lr}
    print(accu_best)
    print(param_best)



