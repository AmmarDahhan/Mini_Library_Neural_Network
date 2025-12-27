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

# يمكننا لاحقاً إضافة محسنات أخرى هنا مثل Adam
# class Adam:
#     def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
#         ...
#     def update(self, params, grads):
#         ...
