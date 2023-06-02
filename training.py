import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch

from fen_into_graphs import fen_into_graph

import os

class ChessGNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(ChessGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        print("xsize: " ,x.size(), "\nedge_index size: ", edge_index.size())
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, torch.zeros(x.size(0), dtype=torch.long), x.size(0))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# There will be 5 categories for now:
# mateIn1, pin, exposedKing, hangingPiece and fork
# respectively 0, 1, 2, 3 and 4 for our GNN
categories = ["matein1", "pin", "exposedKing", "hangingPiece", "fork"]

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

# file_path = os.path.join('Sets', 'training_set')
file_path = "smotheredMate"
train_dataset = ChessTrainDataset(file_path)

graph_train_loader = DataLoader(train_dataset, batch_size=130, shuffle=True, collate_fn=custom_collate)

# graph_val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)  # Additional loader for a larger datasets
# graph_test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

iterator = iter(graph_train_loader)
batch = next(iterator)
print("len: ", len(batch))
batch = Batch.from_data_list(batch)
print("Batch:", batch)
print("Labels:", batch[1].y)
print("Batch indices:", batch.batch[66])
print("next: ", len(batch))
batch = Batch.from_data_list(next(iterator))
print("Batch edge_index:", batch.edge_index)
print("Labels:", batch[1].y)
print("Batch indices:", batch.batch[66])




print("------ Finished reading file, starting the training ------")
num_node_features = 8  # Number of node features (piece and team)
hidden_channels = 32  # Number of hidden channels in GNN layers
num_classes = 5  # Number of classes for graph classification
model = ChessGNN(num_node_features, hidden_channels, num_classes)

# Optimizer definition
optimizer = Adam(model.parameters(), lr=0.01)

# Loss function
criterion = nn.NLLLoss()

# Training the model
model.train()
f = open(os.path.join('trained_models', 'results.txt'), "w+")
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    total_samples = 0
    for x, edge_index, y in graph_train_loader:
        optimizer.zero_grad()
        x_padded = pad_sequence(x, batch_first=True)
        edge_index_list = []
        max_index = x_padded.size(1) - 1  # Initialize the maximum index as the last index of x_padded
        for e in edge_index:
            edge_index_list.append(e + max_index)  # Add the maximum index to adjust the indices
            max_index += x_padded.size(1)  # Update the maximum index
        edge_index_padded = torch.cat(edge_index_list, dim=1)

        output = model(x_padded, edge_index_padded)  # Update the model forward call
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
    avg_loss = total_loss / total_samples
    f.write(f"Epoch {epoch + 1} - Loss: {avg_loss}\n")

print("\n\n------ Finished training, starting the evaluation ------")

# Evaluating the model
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for x, edge_index, y in graph_train_loader:
        x_padded = pad_sequence(x, batch_first=True)
        edge_index_list = []
        max_index = x_padded.size(1) - 1  # Initialize the maximum index as the last index of x_padded
        for e in edge_index:
            edge_index_list.append(e + max_index)  # Add the maximum index to adjust the indices
            max_index += x_padded.size(1)  # Update the maximum index
        edge_index_padded = torch.cat(edge_index_list, dim=1)

        output = model(x_padded, edge_index_padded)  # Update the model forward call
        predicted_labels = output.argmax(dim=1)
        total_correct += (predicted_labels == y).sum().item()
        total_samples += y.size(0)
    accuracy = total_correct / total_samples
    f.write(f"Accuracy: {accuracy}\n")

# Save the model
save_path = os.path.join('trained_models', 'trained_model.pt')
torch.save(model.state_dict(), save_path)
print(f"Model saved at {save_path}")
