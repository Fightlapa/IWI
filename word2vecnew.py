# Dependencies
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Generating desired pair of inputs and targets
# Empty list that collects input and target in pairs
inp_target_list = []

for i in range(26):
    temp = []
    a, b, c, d = i - 2, i - 1, i + 1, i + 2  # targets for input i
    temp.extend([a, b, c, d])
    # keep targets within range of 0 to 25
    for j in range(4):
        if temp[j] >= 0 and temp[j] <= 25:
            inp_target_list.append([i, temp[j]])
print(inp_target_list[:5])
# [[0, 1], [0, 2], [1, 0], [1, 2], [1, 3]]
# Get one hot vectors for all inputs
# Initiate tensor with 0’s that holds all inputs in inp_target pairs
inp_tensor = torch.zeros(len(inp_target_list), 26)
# Substitute 1 for 0, at position indicated by respective input
for i in range(len(inp_tensor)):
    inp_tensor[i, np.array(inp_target_list)[i, 0]] = 1
# One_hot for 0 or letter 'a'
print(inp_tensor[0])
# tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0.])
# Create Network
# 2 fully connected layers with NO bias
# Embedding dimension is 10
# Softmax is implemented using loss criterion (nn.CrossEntropyLoss())
fc1 = nn.Linear(26, 10, bias=False)
fc2 = nn.Linear(10, 26, bias=False)
params = list(fc1.parameters()) + list(fc2.parameters())
LR = 0.001  # Learning rate
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params, lr=LR)
# Train
# Define inputs and target tensors
inp_tensor = inp_tensor
target_tensor = torch.tensor(inp_target_list)[:, 1]
losses = []
for i in range(10000):
    out_1 = fc1(torch.Tensor(inp_tensor))  # hidden layer
    out_2 = fc2(out_1)  # Score matrix
    optimizer.zero_grad()  # Flushing gradients

    loss = criterion(out_2, target_tensor.long().view(out_2.shape[0], ))  # Apply Softmax, get loss

    loss.backward()  # Getting grads
    optimizer.step()  # Correcting parameters
    if i % 1000 == 0:
        losses.append(loss.item())
        print(loss.data)