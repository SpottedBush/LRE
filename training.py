from fen_into_graphs import fen_into_graph
from homemade_GNN import *
import os

# There will be 5 categories for now :
# mateIn1, pin, exposedKing, hangingPiece and fork
# respectively 0, 1, 2, 3 and 4 for our GNN

file = open(os.path.join('Sets', 'training_set'))
tensors = []
y_arr = []
x_arr = [] # [[id node, team, pawn, knight, bishop, rook, queen, king]]
for line in file:
    last_coma = 0
    fen = line[6:]
    idx = 0
    while fen[idx] != ",":
        idx += 1
    fen = fen[0:idx]
    res = fen_into_graph(fen)
    tensors.append(res[0])
    idx += 1
    while line[idx] != "\n":
        if line[idx] == ",":
            last_coma = idx
        idx += 1
    y_arr.append(line[last_coma + 2:])
    x_arr = res[1]
        
        
num_node_features = 8  # Number of node features (piece and team)
hidden_channels = 32  # Number of hidden channels in GCN layers
num_classes = 5 # Number of classes for graph classification
model = ChessGNN(num_node_features, hidden_channels, num_classes)

x_arr = torch.tensor(x_arr, dtype=torch.float32)


x = torch.tensor(x_arr).clone().detach() # Node features
edge_index = torch.tensor(tensors[0], dtype=torch.long) # Edge indices
edge_attr = torch.ones(edge_index.size(1))  # Edge attributes
batch = torch.tensor([0 for i in range(64)])  # Batch indices for graph classification
y = torch.tensor(y_arr) # Labels for graph classification
y = y.expand(64)  # Expanding the scalar label to match batch size 64

# Optimizer definition
optimizer = Adam(model.parameters(), lr=0.01)

# Loss function
criterion = nn.NLLLoss()

# Training the model
model.train()
f = open(os.path.join('trained_models','results.txt'))
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x, edge_index, edge_attr)
    loss = criterion(output[batch], y)
    loss.backward()
    optimizer.step()

    f.write(f"Epoch {epoch+1} - Loss: {loss.item()}")

# Evaluating the model
model.eval()
with torch.no_grad():
    output = model(x, edge_index, edge_attr)
    predicted_labels = output.argmax(dim=1)
    accuracy = (predicted_labels == y).sum().item() / len(y)
    f.write(f"Accuracy: {accuracy}")


# Sauvegarde du model
save_path = os.path.join('trained_models','trained_model.pt')

state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': num_epochs,
    'accuracy': accuracy
}

torch.save(state, save_path)