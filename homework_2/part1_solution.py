# Don't erase the template code, except "Your code here" comments.

import torch
import math                      # Pi

""" Task 1 """

def get_rho():
    # (1) Your code here; theta = ...
    theta = torch.linspace(-math.pi, math.pi, 1000, dtype=torch.float64)
    assert theta.shape == (1000,)

    # (2) Your code here; rho = ...
    rho = (1 + .9 * torch.cos(8*theta)) * (1 + .1 * torch.cos(24*theta)) * (.9 + .05 * torch.cos(200*theta)) * (1 + torch.sin(theta))
    assert torch.is_same_size(rho, theta)
    
    # (3) Your code here; x = ...
    # (3) Your code here; y = ...
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)

    return x, y

""" Task 2 """

def game_of_life_update_torch(alive_map):
    """
    PyTorch version of `game_of_life_update_reference()`.
    
    alive_map:
        `torch.tensor`, ndim == 2, dtype == `torch.int64`
        The game map containing 0s (dead) an 1s (alive).
    """

    # Reshape input for convolution procedure
    old_shape = alive_map.shape
    alive_map = alive_map.view([1, 1] + list(old_shape))

    # Count neighbours for each cell with convolution
    weight = torch.ones((1, 1, 3, 3), dtype=torch.long)
    num_alive_neighbors = (torch.conv2d(alive_map, weight, padding=1) - alive_map).view(old_shape)
    
    # Prepare new alive map
    alive_map = alive_map.view(old_shape)
    new_alive_map = torch.empty_like(alive_map)

    # Update map for dead cells
    new_alive_map[alive_map == 0] = (num_alive_neighbors[alive_map == 0] == 3).long()

    # Update map for alive cells
    new_alive_map[alive_map == 1] = torch.logical_or(num_alive_neighbors[alive_map == 1] == 2,
                                                     num_alive_neighbors[alive_map == 1] == 3).long()

    alive_map.copy_(new_alive_map)

""" Task 3 """

# This is a reference layout for encapsulating your neural network. You can add arguments and
# methods if you need to. For example, you may want to add a method `do_gradient_step()` that
# executes one step of an optimization algorithm (SGD / Adadelta / Adam / ...); or you can
# add an extra argument for which you'll find a good value during experiments and set it as
# default to preserve interface (e.g. `def __init__(self, num_hidden_neurons=100):`).


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, weights):
        with torch.no_grad():
          for layer_weights in weights:
              layer_weights -= self.lr * layer_weights.grad

    def zero_grad(self, weights):
        for layer_weights in weights:
            layer_weights.grad = torch.zeros_like(layer_weights, requires_grad=True)


class Linear:
    def __init__(self, n_in, n_out):
        self.w = torch.randn(n_out, n_in) / 100
        self.b = torch.randn(n_out) / 100
        self.w.requires_grad_(True)
        self.b.requires_grad_(True)

    def forward(self, x):
        return x @ self.w.T + self.b

    def parameters(self):
        return [self.w, self.b]


class ReLU:
    def __init__(self):
        pass
    
    def forward(self, x):
        return torch.maximum(x, torch.tensor(0))

    def parameters(self):
        return []


class NeuralNet:
    def __init__(self):
        # Your code here
        self.linear1 = Linear(784, 100)
        self.activation1 = ReLU()
        self.linear2 = Linear(100, 10)

        self.layers = [self.linear1, self.activation1, self.linear2]

        self.params = []
        for layer in self.layers:
            self.params += layer.parameters()

    def forward(self, x):
        batch_size, height, width = x.shape
        x = x.view(batch_size, height * width)

        x = self.linear1.forward(x)
        x = self.activation1.forward(x)
        x = self.linear2.forward(x)
        return x

    def predict(self, images):
        """
        images:
            `torch.tensor`, shape == `batch_size x height x width`, dtype == `torch.float32`
            A minibatch of images -- the input to the neural net.
        
        return:
        prediction:
            `torch.tensor`, shape == `batch_size x 10`, dtype == `torch.float32`
            The scores of each input image to belong to each of the dataset classes.
            Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
            belong to `j`-th class.
            These scores can be 0..1 probabilities, but for better numerical stability
            they can also be raw class scores after the last (usually linear) layer,
            i.e. BEFORE softmax.
        """
        # Your code here
        batch_size, height, width = images.shape
        images = images.view(batch_size, height * width)

        x = self.linear1.forward(images)
        x = self.activation1.forward(x)
        return self.linear2.forward(x)


def accuracy(model, images, labels):
    """
    Use `NeuralNet.predict` here.
    
    model:
        `NeuralNet`
    images:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    labels:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `images`.
    
    return:
    value:
        `float`
        The fraction of samples from `images` correctly classified by `model`.
        `0 <= value <= 1`.
    """
    # Your code here
    logits = model.predict(images)
    prediction = logits.argmax(dim=1)
    return ((prediction == labels).sum() / labels.shape[0]).item()
    
    
def get_batches(X, y, batch_size):
    n_samples = X.shape[0]
        
    # Shuffle at the start of epoch
    indices = torch.randperm(n_samples)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        batch_idx = indices[start:end]
    
        yield X[batch_idx], y[batch_idx]


def cross_entropy_loss(input, target):
    exp = torch.exp(input)
    norm = exp.sum(axis=1, keepdim=True)
    softmax = exp / norm

    target_one_hot = torch.zeros_like(input)
    target_one_hot[torch.arange(input.shape[0]), target] = 1

    loss = -torch.log(softmax) * target_one_hot
    loss = loss.sum(axis=1).mean()

    return loss


def train_on_notmnist(model, X_train, y_train, X_val, y_val):
    """
    Update `model`'s weights so that its accuracy on `X_val` is >=82%.
    `X_val`, `y_val` are provided for convenience and aren't required to be used.
    
    model:
        `NeuralNet`
    X_train:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    y_train:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `X_train`.
    X_val, y_val:
        Same as above, possibly with a different length.
    """
    # Your code here
    n_epoch = 10
    learning_rate = 1e-1
    optimizer = SGD(lr=learning_rate)
    batch_size = 128
    criterion = cross_entropy_loss

    for epoch in range(n_epoch):
        for X_train_batch, y_train_batch in get_batches(X_train, y_train, batch_size):

            optimizer.zero_grad(model.params)

            output = model.forward(X_train_batch)
            loss = criterion(output, y_train_batch)

            loss.backward()

            optimizer.step(model.params)

        # with torch.no_grad():
        output = model.forward(X_train)
        loss_train = criterion(output, y_train)
        acc_train = accuracy(model, X_train, y_train)

        output = model.forward(X_val)
        loss_val = criterion(output, y_val)
        acc_val = accuracy(model, X_val, y_val)

        print('Epoch {}. Train: loss={}, accuracy={}; Validation: loss={}, accuracy={}'.format(epoch+1, round(loss_train.item(), 2), round(acc_train, 2), round(loss_val.item(), 2), round(acc_val, 2))) 
