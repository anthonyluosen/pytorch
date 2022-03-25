import torch

x = torch.rand(16, 20)
y = torch.randint(2, (16,))
print(x.shape)
print(y.shape)

# Try torch.ones(16) here and it will be equivalent to
# regular CrossEntropyLoss
weights = torch.rand(16)

net = torch.nn.Linear(20, 2)

def element_weighted_loss(y_hat, y, weights):
    m = torch.nn.LogSoftmax(dim=1)
    criterion = torch.nn.MSELoss(reduction='none')
    loss = criterion(m(y_hat), y)
    loss = loss * weights
    return loss.sum() / weights.sum()

weighted_loss = element_weighted_loss(net(x), y, weights)

not_weighted_loss = torch.nn.CrossEntropyLoss()(net(x), y)

print(weighted_loss, not_weighted_loss)