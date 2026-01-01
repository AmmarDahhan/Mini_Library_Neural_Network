import numpy as np 

class SGD:
    def __init__(self,learning_rate=0.01):
        self.lr = learning_rate

    def apply_updates(self,network_parameters,calculate_gradients):
        for key in network_parameters.keys():
            network_parameters[key]-= self.lr*calculate_gradients[key]

class Adam:
    def __init__(self,learning_rate=0.001,beta1=0.9,beta2=0.999):
        self.lr= learning_rate
        self.beta1= beta1
        self.beta2= beta2
        self.step= 0
        self.m= None
        self.v= None

    def apply_updates(self,network_parameters,calculate_gradients):
        if self.m is None:
            self.m,self.v= {},{}
            for key,val in network_parameters.items():
                self.m[key]= np.zeros_like(val)
                self.v[key]= np.zeros_like(val)

        self.step+= 1

        for key in network_parameters.keys():
            grad= calculate_gradients[key]
            self.m[key]= self.beta1*self.m[key] + (1- self.beta1)*grad
            self.v[key]= self.beta2*self.v[key] + (1- self.beta2)*(grad**2)
            m_corrected= self.m[key]/ (1- self.beta1**self.step)
            v_corrected= self.v[key]/ (1- self.beta2**self.step)

            update_value= self.lr*m_corrected/ (np.sqrt(v_corrected)+ 1e-7)
            network_parameters[key]-= update_value
            
        