import numpy as np
import torch
class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.params = dict()
        self.grads = dict()
        self.in_features = in_features
        self.out_features = out_features
        w = torch.empty(size=(out_features,in_features))
        self.params['weight']=torch.nn.init.xavier_normal_(w).numpy()
        self.params['bias'] = np.zeros(out_features)

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        self.input  = np.expand_dims(np.array(x),axis=0)
        out = np.matmul(self.params['weight'],x)+self.params['bias']
        # out = ReLU().forward(out)
        self.out = out
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        # grad computation
        self.grads['weight'] = np.matmul(np.expand_dims(np.array(dout),axis=1),self.input)
        self.grads['bias'] = np.matmul(dout,np.identity(self.out_features))
        dx = np.matmul(dout,self.params['weight'])
        return dx

class ReLU(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        out = np.maximum(x,0)
        # grad varies with the condition of input, so record it
        self.grad = np.zeros((len(x),len(x)))
        for i in range(len(x)):
            if x[i] > 0:
                self.grad[i][i] = 1
        self.out = out
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        # grad computation
        dx = np.matmul(dout,self.grad)
        return dx

class SoftMax(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        m = np.max(x)
        y = np.exp(x-m)
        out = y/np.sum(y)
        self.input = x
        self.out = out
        return out

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        # grad computation
        grad = np.zeros((len(self.out),len(self.input)))
        for i in range(len(self.out)):
            for j in range(len(self.input)):
                if i==j:
                    grad[i][j] = self.out[i]-self.out[i]*self.out[i]
                else:
                    grad[i][j] = -1*self.out[i]*self.out[j]
        dx = np.matmul(dout,grad)
        return dx

class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        logx = np.log(x)
        ylogx = np.dot(y,logx)
        out = -1*np.sum(ylogx)
        self.out = out
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        dx = np.zeros(y.shape)
        for i in range(len(y)):
            dx[i] = -1*(y[i]/x[i])
        return dx
