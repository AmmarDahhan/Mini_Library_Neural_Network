import numpy as np
from collections import OrderedDict
from layers import Affine,Relu,SoftMaxWithLoss

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_type= 'std',l2_penalty= 0):
        self.network_params = {}
        init_type= weight_init_type.lower()

        if init_type== 'he':
            scale1= np.sqrt(2.0/ input_size)
            scale2= np.sqrt(2.0/ hidden_size)
        else:
            scale1= 0.01
            scale2= 0.01

        self.network_params['W1']= scale1 * np.random.randn(input_size,hidden_size)
        self.network_params['b1']= np.zeros(hidden_size)
        self.network_params['W2']= scale2 * np.random.randn(hidden_size,output_size)
        self.network_params['b2']= np.zeros(output_size)

        self.layer_stack= OrderedDict()
        self.layer_stack['Affine1']= Affine(self.network_params['W1'],self.network_params['b1'])
        self.layer_stack['Relu1']= Relu()
        self.layer_stack['Affine2']= Affine(self.network_params['W2'],self.network_params['b2'])

        self.final_loss_layer= SoftMaxWithLoss()
        self.l2_penalty_strength= l2_penalty

    def predict(self,input_data):
        current_data= input_data
        for layer in self.layer_stack.values():
            current_data= layer.forward(current_data)
            
        return current_data
        
    def calculate_loss(self,input_data,target_labels):
        prediction_output = self.predict(input_data)
        l2_penalty_value= 0

        if self.l2_penalty_strength >0:
            w1_penalty= 0.5 * self.l2_penalty_strength * np.sum(self.network_params['W1']**2)
            w2_penalty= 0.5 * self.l2_penalty_strength * np.sum(self.network_params['W2']**2)
            l2_penalty_value= w1_penalty+ w2_penalty

        return self.final_loss_layer.forward(prediction_output,target_labels) + l2_penalty_value
    
    def calculate_accuracy(self,input_data,target_labels):
        predicted_scores= self.predict(input_data)
        predicted_classes= np.argmax(predicted_scores,axis=1)

        if target_labels.ndim !=1:
            target_labels =  np.argmax(target_labels,axis=1)

        accuracy= np.sum(predicted_classes== target_labels)/ float(input_data.shape[0])

        return accuracy
    
    def calculate_gradients(self,input_data,target_labels):
        self.calculate_loss(input_data,target_labels)
        current_grad= self.final_loss_layer.backward(1)
        layers= list(self.layer_stack.values())
        layers.reverse()

        for layer in layers:
            current_grad= layer.backward(current_grad)

        final_gradients= {}
        final_gradients['W1']= self.layer_stack['Affine1'].grad_w + self.l2_penalty_strength* self.network_params['W1']
        final_gradients['b1']= self.layer_stack['Affine1'].grad_b
        final_gradients['W2']= self.layer_stack['Affine2'].grad_w + self.l2_penalty_strength* self.network_params['W2']
        final_gradients['b2']= self.layer_stack['Affine2'].grad_b

        return final_gradients
