from fen_into_graphs import fen_into_graph
from homemade_GNN import *
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_sparse import coalesce, SparseTensor

# There will be 5 categories for now:
# mateIn1, pin, exposedKing, hangingPiece and fork
# respectively 0, 1, 2, 3 and 4 for our GNN
categories = ["matein1", "pin", "exposedKing", "hangingPiece", "fork"]

def strtoidx(str):
    for i in range(len(categories)):
        if str == categories[i]:
            return i
    return 0

class ChessDataset(Dataset):
    def __init__(self, file_path):
        self.tensors = []
        self.y_arr = []
        self.x_arr = [] # [[id node, team, pawn, knight, bishop, rook, queen, king]]
        file = open(file_path)
        for line in file:
            last_coma = 0
            fen = line[6:]
            idx = 0
            while fen[idx] != ",":
                idx += 1
            fen = fen[0:idx]
            res = fen_into_graph(fen)
            self.tensors.append(res[0])
            idx += 1
            while line[idx] != "\n":
                if line[idx] == ",":
                    last_coma = idx
                idx += 1
            self.y_arr.append(strtoidx(line[last_coma + 2:]))
            self.x_arr.append(res[1])

    def __len__(self):
        return len(self.y_arr)

    def __getitem__(self, index):
        x = torch.tensor(self.x_arr[index], dtype=torch.float32)
        edge_index = torch.tensor(self.tensors[index], dtype=torch.long)
        edge_attr = torch.ones(edge_index.size(1))
        y = torch.tensor(self.y_arr[index])
        return x, edge_index, edge_attr, y

file_path = os.path.join('Sets', 'training_set')
dataset = ChessDataset(file_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: zip(*batch))

print("------ Finished reading file, starting the training ------")
num_node_features = 8  # Number of node features (piece and team)
hidden_channels = 32  # Number of hidden channels in GCN layers
num_classes = 5 # Number of classes for graph classification
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
    for x, edge_index, edge_attr, y in dataloader:
        optimizer.zero_grad()
        x_padded = pad_sequence(x, batch_first=True)
        edge_attr_padded = pad_sequence(edge_attr, batch_first=True)
        edge_index_list = []
        offset = 0
        for e in edge_index:
            edge_index_list.append(e + offset)
            offset += x_padded.size(1)
        edge_index_padded = torch.cat(edge_index_list, dim=1)
        edge_attr_padded = edge_attr_padded.view(-1)
        num_nodes = x_padded.size(1)
        edge_index_sparse = SparseTensor(row=edge_index_padded[0], col=edge_index_padded[1], value=edge_attr_padded, sparse_sizes=(num_nodes, num_nodes))
        output = model(x_padded, edge_index_sparse)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    f.write(f"Epoch {epoch+1} - Loss: {avg_loss}\n")

# Evaluating the model
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for x, edge_index, edge_attr, y in dataloader:
        x_padded = pad_sequence(x, batch_first=True)
        edge_attr_padded = pad_sequence(edge_attr, batch_first=True)
        edge_index_list = []
        offset = 0
        for e in edge_index:
            edge_index_list.append(e + offset)
            offset += x_padded.size(1)
        edge_index_padded = torch.cat(edge_index_list, dim=1)
        edge_attr_padded = edge_attr_padded.view(-1)
        num_nodes = x_padded.size(1)
        edge_index_sparse = SparseTensor(row=edge_index_padded[0], col=edge_index_padded[1], value=edge_attr_padded, sparse_sizes=(num_nodes, num_nodes))
        output = model(x_padded, edge_index_sparse)
        predicted_labels = output.argmax(dim=1)
        total_correct += (predicted_labels == y).sum().item()
        total_samples += y.size(0)
    accuracy = total_correct / total_samples
    f.write(f"Accuracy: {accuracy}\n")

# Save the model
save_path = os.path.join('trained_models', 'trained_model.pt')
state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch + 1,
    'accuracy': accuracy
}
torch.save(state, save_path)