import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import seaborn as sns
import matplotlib.pyplot as plt

cuda = True if torch.cuda.is_available() else False

df = pd.read_csv('synthetic_over20.csv')
df2 = pd.read_csv('OTDR_over20_val.csv')
#To stop it going out of bounds
df['Class'] = df['Class'].astype('int')

# Features and target for the original dataset
X = df.drop(columns='Class')
y = df['Class']

#%%

if cuda:
   device = torch.device("cuda")

class GRU_AE(nn.Module):
    def __init__(self, input_size, hidden_sizes, fc_size, num_classes):
        super(GRU_AE, self).__init__()

        # Encoder
        self.encoder_gru1 = nn.GRU(input_size, hidden_sizes[0], batch_first=True)
        self.encoder_gru2 = nn.GRU(hidden_sizes[0], hidden_sizes[1], batch_first=True)

        # Decoder
        self.decoder_gru1 = nn.GRU(hidden_sizes[1], hidden_sizes[0], batch_first=True)
        self.decoder_gru2 = nn.GRU(hidden_sizes[0], fc_size, batch_first=True)

        self.fc = nn.Linear(fc_size, num_classes)

    def forward(self, x):
        # Encoder
        x, _ = self.encoder_gru1(x)
        x, _ = self.encoder_gru2(x)

        # Decoder
        x, _ = self.decoder_gru1(x)
        x, _ = self.decoder_gru2(x)
        x = self.fc(x)

        # Swap dimensions for Cross Entropy Loss
        x = x.permute(0, 2, 1)

        return x

# Hyperparameters
input_size = 32 # Sequence of power levels of length 30
hidden_sizes = [30, 15] # Hidden sizes for the two GRU layers
fc_size = 16 # Fully connected layer size
output_size = 6

# Initialize the model, loss, and optimizer
model = GRU_AE(input_size, hidden_sizes, fc_size, output_size)
criterion = nn.CrossEntropyLoss()

#model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def load_dataset(df, df2):

    # Features and target for the original dataset
    X = df.drop(columns='Class')
    y = df['Class']

    # Stratified split based on the combined key
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

    # Further split the temporary set into validation and initial test sets
    X_val, X_test_initial, y_val, y_test_initial = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Features and target for the holdout set
    X_test = df2.drop(columns='Class')
    y_test = df2['Class']

    # Reshape the datasets
    X_train_reshaped = X_train.values.reshape(X_train.shape[0], 32, 1)
    X_test_reshaped = X_test.values.reshape(X_test.shape[0], 32, 1)
    X_val_reshaped = X_val.values.reshape(X_val.shape[0], 32, 1)

    # Convert data to PyTorch tensors with correct shape
    X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32).permute(0,2,1)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.int64)
    X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32).permute(0,2,1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.int64)
    X_val_tensor = torch.tensor(X_val_reshaped, dtype=torch.float32).permute(0,2,1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.int64)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor

X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = load_dataset(df, df2)

# Create a Dataset from the tensors
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# Define batch size
batch_size = 64

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs=20):
    train_losses = []
    val_losses = []

    # Training loop
    epochs=20
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0  # To compute average training loss
        for batch_X, batch_y in train_loader:
            #batch_X, batch_y = batch_X.type(FloatTensor), batch_y.type(FloatTensor)
            batch_y = batch_y.unsqueeze(-1)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():  # No gradients during validation
            for batch_X, batch_y in val_loader:
                #batch_X, batch_y = batch_X.type(FloatTensor), batch_y.type(FloatTensor)
                batch_y = batch_y.unsqueeze(-1)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                # Compute accuracy
                _, predicted = outputs.max(1)
                val_accuracy += (predicted == batch_y).float().mean().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Plotting the training and validation losses
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

metrics = train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer)

# Testing
model.eval()
correct = 0
total = 0
all_predictions = []
all_true_labels = []

with torch.no_grad():
     for batch_X, batch_y in test_loader:
         #batch_X, batch_y = batch_X.type(FloatTensor), batch_y.type(FloatTensor)
         outputs = model(batch_X)
         predicted = torch.argmax(outputs, dim=1).squeeze()

         all_predictions.extend(predicted.cpu().numpy())
         all_true_labels.extend(batch_y.cpu().numpy())

         correct += (predicted == batch_y).sum().item()
         total += batch_y.numel()

         accuracy = 100 * correct / total

     print(f"Test Accuracy: {accuracy:.2f}%")

     # Classification report
     report = classification_report(all_true_labels, all_predictions, output_dict=True, zero_division=1)
     for class_label, metrics in report.items():
          if class_label.isdigit():
             print(f"Class {class_label}:")
             print(f"\tPrecision: {metrics['precision']}")
             print(f"\tRecall: {metrics['recall']}")
             print(f"\tF1-score: {metrics['f1-score']}\n")


def plot_confusion_matrix(true_labels, predicted_labels, classes):
    """
    Plot a confusion matrix using ground truth and predictions.

    :param true_labels: List of true labels
    :param predicted_labels: List of predicted labels
    :param classes: List of class names
    """
    # Compute confusion matrix
    matrix = confusion_matrix(true_labels, predicted_labels)

    # Create a heatmap
    plt.figure(figsize=(13, 9))
    sns.set(font_scale=1.2)
    sns.heatmap(matrix, annot=True, fmt="g", cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(all_true_labels, all_predictions, classes=['PC Connector', 'Normal', 'Bad Splice', 'Reflector', 'Fiber Tapping', 'Dirty Connector'])

num_classes = len(np.unique(all_true_labels))

# Convert all_true_labels and all_predictions to one-hot encoded form for multi-class AUCPR calculation
true_labels_one_hot = np.eye(num_classes)[all_true_labels]
predictions_one_hot = np.eye(num_classes)[all_predictions]

# Calculate AUCPR for each class
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(true_labels_one_hot[:, i], predictions_one_hot[:, i])
    aucpr = auc(recall, precision)
    print(f"Class {i} AUCPR: {aucpr:.4f}")

target_class = 4  # Replace with your target class label
num_classes = len(np.unique(all_true_labels))  # Adjust based on your number of classes

# Convert all_true_labels and all_predictions to one-hot encoded form
true_labels_one_hot = np.eye(num_classes)[all_true_labels]
predictions_one_hot = np.eye(num_classes)[all_predictions]

# Calculate precision and recall for the target class
precision, recall, _ = precision_recall_curve(true_labels_one_hot[:, target_class], predictions_one_hot[:, target_class])

# Plotting the Precision-Recall curve
plt.figure()
plt.plot(recall, precision, label=f'Upper Bound Baseline')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve for Class {target_class}')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()