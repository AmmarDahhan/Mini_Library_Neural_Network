import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from network import TwoLayerNet
from optimizer import SGD

# 1. تحميل ومعالجة البيانات (MNIST)
print("Loading dataset...")
# جلب البيانات
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# تطبيع قيم البكسلات لتكون بين 0 و 1
X = X / 255.0

# تحويل التسميات (y) من نص إلى أرقام
y = y.astype(np.uint8)

# تحويل التسميات إلى صيغة "one-hot"
# مثال: الرقم 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
enc = OneHotEncoder(sparse_output=False, categories='auto')
y_one_hot = enc.fit_transform(y[:, np.newaxis])

# تقسيم البيانات إلى مجموعة تدريب ومجموعة اختبار
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y_one_hot[:60000], y_one_hot[60000:]
y_train_labels, y_test_labels = y[:60000], y[60000:] # للاستخدام في حساب الدقة

print("Dataset loaded and preprocessed.")

# 2. تعريف المعلمات الفائقة (Hyperparameters)
iters_num = 10000       # عدد مرات تكرار التدريب
train_size = X_train.shape[0]
batch_size = 100        # حجم الدفعة الصغيرة في كل مرة تدريب
learning_rate = 0.1     # معدل التعلم

# 3. تهيئة الشبكة والمحسن
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
optimizer = SGD(lr=learning_rate)

print("Starting training...")

# 4. حلقة التدريب الرئيسية
for i in range(iters_num):
    # أ. اختيار دفعة صغيرة عشوائية (mini-batch)
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = X_train[batch_mask]
    t_batch = y_train[batch_mask]
    
    # ب. حساب التدرجات (الانتشار الأمامي والخلفي)
    grads = network.gradient(x_batch, t_batch)
    
    # ج. تحديث الأوزان
    optimizer.update(network.params, grads)
    
    # كل 100 خطوة، اطبع حالة التدريب
    if i % 100 == 0:
        # حساب الخسارة والدقة على مجموعة التدريب والاختبار
        train_loss = network.loss(x_batch, t_batch)
        
        # حساب الدقة
        train_acc = network.accuracy(X_train, y_train)
        test_acc = network.accuracy(X_test, y_test)
        
        print(f"Iteration {i}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

print("Training finished.")

# 5. تقييم نهائي
final_train_acc = network.accuracy(X_train, y_train)
final_test_acc = network.accuracy(X_test, y_test)
print("="*30)
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Test Accuracy: {final_test_acc:.4f}")

