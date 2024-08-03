import torch.nn as nn
import torch.optim as optim
import torch.nn.init
import torch.nn.functional as F
from data_load import *
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(4*4*10, 80)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    


def accuracy(predictions, truth):
    predicted_labels = torch.argmax(predictions, axis=1)
    correct = (predicted_labels == truth).float()
    accuracy = correct.mean().item()
    return accuracy


model = CNN()

learning_rate = 0.01
batch_size = 55
epochs = 15

lossfunc = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



loss_values = []
acc_values = []

for epoch in range(epochs):
    for i, traindata in enumerate(train_dataloader):
        img, label = traindata 
       
        prediction = model.forward(img)

        
        loss = lossfunc(prediction, label)


        optimizer.zero_grad()


        loss.backward()


        optimizer.step()


        acc = accuracy(F.softmax(prediction, dim = 1), label)


    loss_values.append(loss.detach().clone().numpy())
    acc_values.append(acc)

    print(f"Epoch {epoch+1}----------------------------")
    print(f"loss: {loss.item():.6f}")
    print(f"acc: {acc:.6f}")


torch.save(model.state_dict(),'model_weight.pth')




plt.figure()
plt.plot(loss_values, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(acc_values, label="accuracy")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.legend()
plt.grid()
plt.show()