import torch
import torch.nn as nn
from data_load import *
import numpy as np
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
    

model = CNN()
model.load_state_dict(torch.load('model_weight.pth'))




test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

def plot_model_prediction(index):
    model.eval()
    img, true_label = test_data[index]
    img = img.unsqueeze(0)

    with torch.no_grad():
        prediction = model(img)
        predicted_label_index = torch.argmax(prediction, axis=1).item()

    fig, ax = plt.subplots()
    ax.imshow(img.squeeze().numpy(), cmap="gray")
    ax.set_title(f"Predicted: {predicted_label_index}\nTruth: {true_label}")

    return fig, ax

index = np.random.randint(0, len(test_data))
plot_model_prediction(index)
plt.show()