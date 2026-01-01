import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from network import TwoLayerNet
from optimizer import SGD, Adam

CHOSEN_OPTIMIZER = 'sgd'     # sgd or adam
USE_HE_INIT = False            # True or False
L2_PENALTY = 0             # 0 or 1e-4


np.random.seed(42) 


print("Data loading MNIST " )
X_data,y_data= fetch_openml('mnist_784',version=1,return_X_y=True,as_frame=False)
X_data= X_data/ 255.0
y_data_labels= y_data.astype(np.uint8)
encoder= OneHotEncoder(sparse_output=False)
y_data_onehot= encoder.fit_transform(y_data_labels.reshape(-1, 1))
X_train,X_test= X_data[:60000],X_data[60000:]
y_train, y_test= y_data_onehot[:60000],y_data_onehot[60000:]
y_train_labels,y_test_labels= y_data_labels[:60000],y_data_labels[60000:]

print("The data was uploaded successfully")


total_training_steps=10000
training_data_size =X_train.shape[0]
batch_size= 100

print("\nNetwork settings")
print(f"engine (Optimizer): {CHOSEN_OPTIMIZER}")
print(f"Using the configuration He: {USE_HE_INIT}")
print(f"Penalty L2: {L2_PENALTY}")
print("\n")

network= TwoLayerNet(input_size=784,hidden_size=50,output_size=10,weight_init_type='he' if USE_HE_INIT else 'std',l2_penalty=L2_PENALTY)

if CHOSEN_OPTIMIZER.lower() == 'adam':
    optimizer= Adam(learning_rate=0.001)
else:
    optimizer= SGD(learning_rate=0.1)

print("Start of training session\n")

for step in range(total_training_steps):
    random_indices= np.random.choice(training_data_size,batch_size)
    input_batch= X_train[random_indices]
    target_batch= y_train[random_indices]   
    gradients= network.calculate_gradients(input_batch,target_batch)
    optimizer.apply_updates(network.network_params,gradients)
    
    if step % 100 == 0:
        current_loss= network.calculate_loss(input_batch,target_batch)
        accuracy_on_train= network.calculate_accuracy(X_train,y_train_labels)
        accuracy_on_test= network.calculate_accuracy(X_test,y_test_labels)
        
        print(f"the step {step}/ {total_training_steps} | the error: {current_loss:.4f} | Training accuracy : {accuracy_on_train:.4f} | Test accuracy: {accuracy_on_test:.4f}")

print("\n Training End" )

final_accuracy= network.calculate_accuracy(X_test,y_test_labels)
print("=" * 40)
print(f"Final accuracy on test data: {final_accuracy:.4f}")
print("=" * 40)
