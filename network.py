import numpy as np
from collections import OrderedDict
from layers import Affine, Relu, SoftmaxWithLoss

class TwoLayerNet:
    
    # A simple two-layer neural network.
    # Architecture: Affine -> ReLU -> Affine -> Softmax
    
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, weight_init_type='std', weight_decay_lambda=0):
    
        # Initializes the network.
        # input_size: حجم المدخلات (مثلاً 784 لصورة 28x28)
        # hidden_size: عدد العصبونات في الطبقة المخفية
        # output_size: عدد المخرجات (مثلاً 10 للأرقام من 0-9)
    
        
        # 1. تهيئة الأوزان والانحيازات (Parameters)
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 2. بناء الطبقات بالترتيب
        # نستخدم OrderedDict لضمان الحفاظ على ترتيب الطبقات
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        # الطبقة الأخيرة منفصلة لأنها تتعامل مع حساب الخسارة
        self.lastLayer = SoftmaxWithLoss()
         # تخزين معامل weight decay
        self.weight_decay_lambda = weight_decay_lambda
        
    def predict(self, x):
    
        # Performs prediction (forward pass without loss calculation).
    
        # تمرير المدخلات عبر كل طبقة بالترتيب
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x

    def loss(self, x, t):
    
        # Calculates the loss.
        # x: input data
        # t: target labels
        
        # نقوم أولاً بالتنبؤ
        y = self.predict(x)
        # حساب عقوبة الأوزان (L2 norm)
        weight_decay = 0
        weight_decay += 0.5 * self.weight_decay_lambda * np.sum(self.params['W1'] ** 2)
        weight_decay += 0.5 * self.weight_decay_lambda * np.sum(self.params['W2'] ** 2)

        # ثم نحسب الخسارة باستخدام الطبقة الأخيرة مع إضافة العقوبة
        return self.lastLayer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        
        # Calculates the accuracy.
        
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: 
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        
        # Calculates gradients for all weights and biases using backpropagation.
        
        # 1. الانتشار الأمامي (Forward pass)
        self.loss(x, t)

        # 2. الانتشار الخلفي (Backward pass)
        dout = 1
        # نبدأ من الطبقة الأخيرة
        dout = self.lastLayer.backward(dout)
        
        # نعكس ترتيب الطبقات للانتشار الخلفي
        layers = list(self.layers.values())
        layers.reverse()
        
        # نمرر "اللوم" (dout) عبر كل طبقة بالعكس
        for layer in layers:
            dout = layer.backward(dout)

         # 3. تجميع التدرجات وإضافة تدرج عقوبة L2
        grads = {}
        # نضيف تدرج العقوبة (lambda * W) إلى تدرج الوزن المحسوب
        grads['W1'] = self.layers['Affine1'].dW + self.weight_decay_lambda * self.params['W1']
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW + self.weight_decay_lambda * self.params['W2']
        grads['b2'] = self.layers['Affine2'].db

        return grads
