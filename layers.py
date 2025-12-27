import numpy as np

class Relu:
    
    # ReLU Layer.
    
    def __init__(self):
        # قناع لتخزين الأماكن التي كانت فيها قيمة المدخلات أقل من أو تساوي صفر
        self.mask = None

    def forward(self, x):
        
        # Forward pass.
        # x: المدخلات (يمكن أن تكون أي مصفوفة numpy)
        
        # القناع سيكون True في الأماكن التي x <= 0, و False في غير ذلك
        self.mask = (x <= 0)
        out = x.copy()  # ننشئ نسخة لتجنب تغيير المدخلات الأصلية
        out[self.mask] = 0  # تصفير القيم غير النشطة
        return out

    def backward(self, dout):
        
        # Backward pass.
        # dout: التدرج القادم من الطبقة التالية
        
        # خلال الانتشار العكسي، التدرج يمر فقط عبر العصبونات التي كانت "نشطة"
        # لذلك، نقوم بتصفير تدرج العصبونات التي كانت "خاملة"
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    
    # Sigmoid Layer.
    
    def __init__(self):
        # سنقوم بتخزين المخرجات (self.out) لأننا نحتاجها في حساب الانتشار العكسي
        self.out = None

    def forward(self, x):
    
        # Forward pass.
    
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
    
        # Backward pass.
        # مشتقة دالة Sigmoid هي out * (1 - out)
    
        # نطبق قاعدة السلسلة: نضرب التدرج القادم (dout) بمشتقة الدالة المحلية
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Affine:
    
    # Affine (Fully Connected) Layer.
    
    def __init__(self, W, b):
        # W: مصفوفة الأوزان, b: متجه الانحياز
        self.W = W
        self.b = b
        
        # سنحفظ المدخلات (x) لحساب الانتشار العكسي
        self.x = None
        # سنحفظ شكل المدخلات الأصلي في حال كانت صورة وتحتاج لإعادة تشكيل
        self.original_x_shape = None
        
        # هذه المتغيرات ستحتوي على تدرج الأوزان والانحياز بعد الحساب
        self.dW = None
        self.db = None

    def forward(self, x):
    
        # Forward pass.
    
        # إذا كانت المدخلات عبارة عن بيانات صور (مثلاً 100 صورة بحجم 28*28)
        # سيتم تحويلها إلى مصفوفة ثنائية الأبعاد (100, 784)
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        
        # تنفيذ العملية الرئيسية: Y = X • W + b
        out = np.dot(self.x, self.W) + self.b
        
        return out

    def backward(self, dout):
    
        # Backward pass.
        # dout: التدرج القادم من الطبقة التالية
    
        # 1. حساب التدرج بالنسبة للمدخلات (dx)، ليتم تمريره للطبقة السابقة
        dx = np.dot(dout, self.W.T)
        
        # 2. حساب التدرج بالنسبة للأوزان (dW)
        self.dW = np.dot(self.x.T, dout)
        
        # 3. حساب التدرج بالنسبة للانحياز (db)
        # الانحياز يطبق على كل عينة في الدفعة (batch), لذا تدرجه هو مجموع تدرجات الدفعة
        self.db = np.sum(dout, axis=0)
        
        # إعادة تشكيل dx ليطابق شكل المدخلات الأصلي قبل تمريره للخلف
        dx = dx.reshape(self.original_x_shape)
        return dx

def softmax(x):
    # خدعة لضمان الاستقرار الرقمي ومنع القيم الكبيرة جداً
    # نطرح أكبر قيمة من كل القيم لمنع حدوث overflow عند حساب الأس
    x = x - np.max(x, axis=-1, keepdims=True)   
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    # y هي مخرجات الشبكة (الاحتمالات)
    # t هي الإجابات الصحيحة (one-hot encoded)
    
    # إذا كانت y عبارة عن دفعة من البيانات، نحسب الخطأ لكل عينة ثم نجمع
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # إذا لم تكن الإجابات الصحيحة one-hot, نحولها
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    # نضيف قيمة صغيرة جداً (1e-7) لتجنب حساب لوغاريتم الصفر (log(0))
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class SoftmaxWithLoss:
    
    # Softmax with Cross Entropy Error Loss Layer.
    
    def __init__(self):
        self.loss = None # سيخزن قيمة الخطأ
        self.y = None    # سيخزن مخرجات الـ softmax (الاحتمالات)
        self.t = None    # سيخزن الإجابات الصحيحة (labels)

    def forward(self, x, t):
    
        # Forward pass.
        # x: المدخلات من الطبقة الأخيرة (Affine)
        # t: الإجابات الصحيحة

        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):

        # Backward pass.
        # dout: عادة ما يكون 1، لأن هذه هي الطبقة التي يبدأ منها حساب التدرج

        batch_size = self.t.shape[0]
        # هذا هو الاختصار الرياضي السحري!
        # التدرج الذي يجب تمريره للخلف هو ببساطة (الناتج المتوقع - الناتج الحقيقي)
        dx = (self.y - self.t) / batch_size
        
        return dx


