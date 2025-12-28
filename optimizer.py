import numpy as np

class SGD:
    
    # Stochastic Gradient Descent (SGD) optimizer.
    
    def __init__(self, lr=0.01):
    
        # Initializes the optimizer.
        # lr: Learning Rate (معدل التعلم). هو الذي يحدد حجم "الخطوة" التي نأخذها لتصحيح الأوزان.
        
        self.lr = lr
        
    def update(self, params, grads):
        
        # Updates parameters based on the calculated gradients.
        # params: قاموس يحتوي على معاملات الشبكة (e.g., {'W1': W1, 'b1': b1, ...})
        # grads: قاموس يحتوي على التدرجات لتلك المعاملات (e.g., {'W1': dW1, 'b1': db1, ...})
        
        for key in params.keys():
            # هذه هي المعادلة الأساسية للتحديث
            # New_Weight = Old_Weight - Learning_Rate * Gradient
            params[key] -= self.lr * grads[key]

class Adam:
    # Adam optimizer
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
