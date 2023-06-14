import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch
from matplotlib.pylab import plt
from numpy import arange

from fen_into_graphs import fen_into_graph

import os

from hyperopt import hp, tpe, fmin


num_nodes = 65
num_node_features = 7  # Number of node features (piece and team)
num_classes = 5  # Number of classes for graph classification
BATCH_SIZE = 130


def matprint(mat):
    print("conf_matrix is: ")
    for column in mat:
        print("|  ", end = "")
        for line in column:
            print(line, end=" | ")
        print("")
    print("\n")


mat = [[1,2,3],[4,5,6],[7,8,9]]
matprint(mat)

def plt_print(train_values):
    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(len(train_values))

    # Plot and label the training and validation loss values
    plt.plot(epochs, train_values, label='Training Loss')
    #plt.plot(epochs, val_values, label='Validation Loss')

    # Add in a title and axes labels
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set the tick locations
    plt.xticks(arange(0, len(train_values), 2))

    # Display the plot
    plt.legend(loc='best')
    plt.show()

class ChessGNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(ChessGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, torch.zeros(x.size(0), dtype=torch.long), x.size(0))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# There will be 5 categories for now:
# mateIn1, pin, mateIn2, hangingPiece and fork
# respectively 0, 1, 2, 3 and 4 for our GNN
categories = ["mateIn1", "pin", "mateIn2", "hangingPiece", "fork"]
model = ChessGNN(num_node_features, 2, num_classes)
def strtoidx(str):
    for i in range(len(categories)):
        if categories[i] in str:
            return i
    return -1

class ChessTrainDataset(Dataset):
    def __init__(self, file_path):
        super(ChessTrainDataset, self).__init__()
        self.data_list = []  # List to store the graph data

        # Read the file and process the data
        with open(file_path, 'r') as file:
            for line in file:
                fen = line.strip().split(',')[1]
                graph_data = fen_into_graph(fen)  # Convert FEN notation to graph data

                # Create a `Data` object and add it to the list
                data = Data(x=graph_data[0], edge_index=graph_data[1], y=strtoidx(line.strip().split(',')[7]), num_node_features = 7, num_nodes =  65)
                self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

def custom_collate(batch):
    return batch

train_path = os.path.join('Sets', 'training_set')
train_dataset = ChessTrainDataset(train_path)
graph_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)


val_path = os.path.join('Sets', 'validation_set')
val_dataset = ChessTrainDataset(val_path)
graph_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)  # Additional loader for a larger datasets

test_path = os.path.join('Sets', 'testing_set')
test_dataset = ChessTrainDataset(test_path)
graph_test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

iterator = iter(graph_train_loader)
batch = next(iterator)


print("\n------ Finished reading file, starting the training ------\n\n")

def train_model(params, num_node_features):
    hidden_channels = params['hidden_channels']
    learning_rate = params['learning_rate']
    num_epochs = params['num_epochs']

    model = ChessGNN(num_node_features, hidden_channels, num_classes)
    optimizer = Adam(model.parameters(), learning_rate)
    criterion = nn.NLLLoss()

    model.train()
    training_values = []
    for epoch in range(num_epochs):
        total_loss = 0
        total_samples = 0
        conf_matrix = [[0 for _ in range(len(categories))] for _ in range(len(categories))]
        for elem in graph_train_loader:
            for x, edge_index, y, num_node_features, num_nodes in elem:
                if y[1] == -1:
                    continue
                actual_label = y[1]
                x = torch.tensor(x[1], dtype=torch.float32)
                edge_index = torch.tensor(edge_index[1])
                edge_index = edge_index.transpose(0, 1)
                y = [y[1] for _ in range(num_nodes[1])]
                y = torch.tensor(y)

                optimizer.zero_grad()
                output = model(x, edge_index)
                loss = criterion(output, y)
                predicted = torch.max(output.data, 1).indices
                predicted_label = round(predicted.float().mean().item())

                conf_matrix[actual_label][predicted_label] += 1
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y.size(0)
                total_samples += y.size(0)
        training_values.append(total_loss)
        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch + 1} - Loss: {avg_loss}\n")

    return {'loss': -avg_loss, 'status': 'ok'}


space = {
    'hidden_channels': hp.choice('hidden_channels', [8, 16, 32]),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'num_epochs': hp.choice('num_epochs', [10, 20, 30])
}

best = fmin(
    fn=lambda params: train_model(params,num_node_features),
    space=space,
    algo=tpe.suggest,
    max_evals=10,
)

print(best)

print("\n------ Finished training, starting the evaluation ------\n\n")

# Evaluating the model
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for elem in graph_test_loader:
        for x, edge_index, y, num_node_features, num_nodes in elem:
            x = torch.tensor(x[1], dtype=torch.float32)
            edge_index = torch.tensor(edge_index[1])
            edge_index = edge_index.transpose(0, 1)
            y = [y[1] for _ in range(num_nodes[1])]
            y = torch.tensor(y)
            output = model(x, edge_index)
            predicted_labels = output.argmax(dim=1)
            total_correct += (predicted_labels == y).sum().item()
            total_samples += y.size(0)
    accuracy = total_correct / total_samples
    print(f"Accuracy: {accuracy}\n")

print(f"Parameters were:\nhidden_channels: {best['hidden_channels']}\n"
      f"learning_rate: {best['learning_rate']}\n"
      f"num_epochs: {best['num_epochs']}")

# Save the model
save_path = os.path.join('trained_models', 'Hyper_opt_model.pt')
torch.save(model.state_dict(), save_path)
print(f"Model saved at {save_path}")