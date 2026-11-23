import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#Hyperparameters
learningRate = 0.001
epochs = 100
#training proportion is remaining data
test_size = 0.15
val_size = 0.15


data = pd.read_csv("Occupancy_Estimation.csv")

X = data.drop(columns=[
    "Date",
    "Time",
    "Room_Occupancy_Count"
])

y = data["Room_Occupancy_Count"]

#data splitting int training, validation, and testing sets, selected randomly

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    stratify=y
)

val_relative_size = val_size / (1 - test_size) 

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=val_relative_size,
    stratify=y_train_val
)

#normalize training data, use sd and mean of training data to normalize testing and validation data

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.long)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

class OccupancyMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.network(x)

model = OccupancyMLP(X_train.shape[1])

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learningRate
)

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):

    model.train()

    outputs = model(X_train)
    train_loss = loss_function(outputs, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = loss_function(val_outputs, y_val)

        val_predictions = torch.argmax(val_outputs, dim=1)
        val_accuracy = (val_predictions == y_val).float().mean()

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    val_accuracies.append(val_accuracy.item())

    if epoch % 10 == 0:
        print(
            "Epoch:", epoch,
            "Train Loss:", train_loss.item(),
            "Val Loss:", val_loss.item(),
            "Val Accuracy:", val_accuracy.item()
        )


model.eval()

with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = torch.argmax(test_outputs, dim=1)

    test_accuracy = (test_predictions == y_test).float().mean()

print("Test Accuracy:", test_accuracy.item())

#training/validation loss graph
plt.figure()
plt.plot(range(epochs), train_losses, label="Training Loss")
plt.plot(range(epochs), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

#validation accuracy graph
plt.figure()
plt.plot(range(epochs), val_accuracies)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy")
plt.ylim(0, 1)
plt.show()
