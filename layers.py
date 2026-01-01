import numpy as np

# Helper Func.

def softmax(input_data):
    stabilized_data = input_data - np.max(input_data,axis=1, keepdims=True)
    exponentails = np.exp(stabilized_data)

    return exponentails / np.sum(exponentails,axis=-1,keepdims=True) 


def cross_entropy_error(predicted_probs,true_labels):
    if predicted_probs.ndim == 1:
        predicted_probs = predicted_probs.reshape(1,predicted_probs.size)
        true_labels = true_labels.reshape(1,true_labels.size)

    if true_labels.ndim== 2:
        true_labels = true_labels.argmax(axis=1)
    batch_size = predicted_probs.shape[0]
    correct_log_probs = np.log(predicted_probs[np.arange(batch_size),true_labels]+ 1e-7)
    loos = -np.sum(correct_log_probs)/batch_size

    return loos

# Layers

class Relu:
    def __init__(self):
        self.zero_mask = None

    def forward(self,input_data):
        self.zero_mask = (input_data <=0)
        output_data = input_data.copy()
        output_data[self.zero_mask]= 0

        return output_data
    
    def backward(self,grad_from_next_layer):
        grad_from_next_layer[self.zero_mask]= 0
        grad_for_prev_layer= grad_from_next_layer

        return grad_for_prev_layer
    
class Sigmoid:
    def __init__(self):
        self.output= None

    def forward(self,input_data):
        result= 1 /(1+ np.exp(-input_data))
        self.output = result
        return result
    
    def backward(self, grad_from_next_layer):
        grade_for_prev_layer= grad_from_next_layer * (1.0 - self.output) * self.output
        return grade_for_prev_layer
    
class Affine:
    def __init__(self,weights,bias):
        self.W= weights
        self.b= bias

        self.input_data = None
        self.grad_w= None
        self.grad_b= None

    def forward(self,input_data):
        self.input_data= input_data
        result = np.dot(input_data,self.W) +self.b
        
        return result
    
    def backward(self,grad_from_next_layer):
        grad_for_prev_layer= np.dot(grad_from_next_layer,self.W.T)
        self.grad_w = np.dot(self.input_data.T,grad_from_next_layer)
        self.grad_b= np.sum(grad_from_next_layer,axis=0)

        return grad_for_prev_layer
    
class SoftMaxWithLoss:
    def __init__(self):
        self.loss = None
        self.predicted_probs= None
        self.true_labels  = None
    
    def forward(self,input_data,true_labels):
        self.true_labels= true_labels
        self.predicted_probs= softmax(input_data)
        self.loss= cross_entropy_error(self.predicted_probs,self.true_labels)

        return self.loss
    
    def backward(self,dout= 1):
        batch_size= self.true_labels.shape[0]
        grad_for_prev_layer= (self.predicted_probs - self.true_labels)/ batch_size

        return grad_for_prev_layer
    
    
