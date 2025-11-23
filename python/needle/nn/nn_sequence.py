"""The module.
"""
from typing import List, Tuple, Optional
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # sigmoid(x) = 1 / (1 + exp(-x))
        return ops.power_scalar(
            ops.add_scalar(ops.exp(ops.negate(x)), 1),
            -1
        )
        ### END YOUR SOLUTION



class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype
        
        # Initialize bound for uniform distribution
        bound = 1.0 / np.sqrt(hidden_size)
        
        # Initialize weights
        self.W_ih = Parameter(init.rand(input_size, hidden_size, 
                                       low=-bound, high=bound, 
                                       device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, 
                                       low=-bound, high=bound, 
                                       device=device, dtype=dtype))
        
        # Initialize biases
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, 
                                              low=-bound, high=bound, 
                                              device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(hidden_size, 
                                              low=-bound, high=bound, 
                                              device=device, dtype=dtype))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        
        # Initialize hidden state if not provided
        if h is None:
            h = init.zeros(batch_size, self.hidden_size, 
                          device=self.device, dtype=self.dtype)
        
        # Compute: h' = tanh(X @ W_ih + bias_ih + h @ W_hh + bias_hh)
        out = X @ self.W_ih + h @ self.W_hh
        
        if self.bias:
            # Broadcast biases
            out = out + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(out.shape)
            out = out + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(out.shape)
        
        # Apply nonlinearity
        if self.nonlinearity == 'tanh':
            return ops.tanh(out)
        elif self.nonlinearity == 'relu':
            return ops.relu(out)
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        
        # Create RNN cells for each layer
        self.rnn_cells = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            cell = RNNCell(layer_input_size, hidden_size, bias, nonlinearity, device, dtype)
            self.rnn_cells.append(cell)
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        ### BEGIN YOUR SOLUTION
        seq_len, batch_size, input_size = X.shape
        
        # Initialize hidden states if not provided
        if h0 is None:
            h0 = init.zeros(self.num_layers, batch_size, self.hidden_size,
                           device=self.device, dtype=self.dtype)
        
        # Split X along sequence dimension
        X_splits = list(ops.split(X, axis=0))
        
        # Split h0 along layer dimension  
        h_splits = list(ops.split(h0, axis=0))
        
        # Process each timestep
        outputs = []
        h_current = [h.reshape((batch_size, self.hidden_size)) for h in h_splits]
        
        for t in range(seq_len):
            # Reshape x_t from (1, bs, input_size) to (bs, input_size)
            x_t = X_splits[t].reshape((batch_size, input_size))
            
            # Process through each layer
            for layer in range(self.num_layers):
                h_current[layer] = self.rnn_cells[layer](x_t, h_current[layer])
                x_t = h_current[layer]  # Output becomes input to next layer
            
            outputs.append(h_current[-1])
        
        # Stack outputs and final hidden states
        output = ops.stack(outputs, axis=0)
        h_n = ops.stack(h_current, axis=0)
        
        return output, h_n
        ### END YOUR SOLUTION

class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        
        bound = 1.0 / np.sqrt(hidden_size)
        
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size,
                                       low=-bound, high=bound,
                                       device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size,
                                       low=-bound, high=bound,
                                       device=device, dtype=dtype))
        
        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size,
                                              low=-bound, high=bound,
                                              device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(4 * hidden_size,
                                              low=-bound, high=bound,
                                              device=device, dtype=dtype))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        
        if h is None:
            h0 = init.zeros(batch_size, self.hidden_size,
                           device=self.device, dtype=self.dtype)
            c0 = init.zeros(batch_size, self.hidden_size,
                           device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h
        
        # Compute gates
        gates = X @ self.W_ih + h0 @ self.W_hh
        
        if self.bias:
            bias_sum = self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(gates.shape) + \
                      self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(gates.shape)
            gates = gates + bias_sum
        
        # Split gates: reshape to (batch_size, 4, hidden_size) then split
        gates_reshaped = gates.reshape((batch_size, 4, self.hidden_size))
        gates_list = list(ops.split(gates_reshaped, axis=1))
        
        # Remove singleton dimension from each gate
        i = Sigmoid()(gates_list[0].reshape((batch_size, self.hidden_size)))
        f = Sigmoid()(gates_list[1].reshape((batch_size, self.hidden_size)))
        g = ops.tanh(gates_list[2].reshape((batch_size, self.hidden_size)))
        o = Sigmoid()(gates_list[3].reshape((batch_size, self.hidden_size)))
        
        # Compute new cell and hidden states
        c_new = f * c0 + i * g
        h_new = o * ops.tanh(c_new)
        
        return h_new, c_new
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        
        # Create LSTM cells for each layer
        self.lstm_cells = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            cell = LSTMCell(layer_input_size, hidden_size, bias, device, dtype)
            self.lstm_cells.append(cell)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        ### BEGIN YOUR SOLUTION
        seq_len, batch_size, input_size = X.shape
        
        # Initialize hidden and cell states if not provided
        if h is None:
            h0 = init.zeros(self.num_layers, batch_size, self.hidden_size,
                           device=self.device, dtype=self.dtype)
            c0 = init.zeros(self.num_layers, batch_size, self.hidden_size,
                           device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h
        
        # Split inputs along sequence and layer dimensions
        X_splits = list(ops.split(X, axis=0))
        h_splits = list(ops.split(h0, axis=0))
        c_splits = list(ops.split(c0, axis=0))
        
        # Reshape hidden and cell states
        h_current = [h.reshape((batch_size, self.hidden_size)) for h in h_splits]
        c_current = [c.reshape((batch_size, self.hidden_size)) for c in c_splits]
        
        # Process each timestep
        outputs = []
        
        for t in range(seq_len):
            # Reshape x_t from (1, bs, input_size) to (bs, input_size)
            x_t = X_splits[t].reshape((batch_size, input_size))
            
            # Process through each layer
            for layer in range(self.num_layers):
                h_current[layer], c_current[layer] = self.lstm_cells[layer](
                    x_t, (h_current[layer], c_current[layer])
                )
                x_t = h_current[layer]  # Output becomes input to next layer
            
            outputs.append(h_current[-1])
        
        # Stack outputs and final states
        output = ops.stack(outputs, axis=0)
        h_n = ops.stack(h_current, axis=0)
        c_n = ops.stack(c_current, axis=0)
        
        return output, (h_n, c_n)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        # Initialize weight matrix from N(0, 1)
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim,
                                          mean=0.0, std=1.0,
                                          device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Memory-efficient embedding using one_hot operation
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        
        # Flatten indices for one_hot creation
        batch_size = seq_len * bs
        indices_flat = x.reshape((batch_size,))
        
        # Use memory-efficient one_hot from init module
        # This creates one-hot vectors one at a time, not a huge matrix
        one_hot = init.one_hot(self.num_embeddings, indices_flat, 
                              device=self.device, dtype=self.dtype)
        
        # Matrix multiply for embedding lookup
        embedded = one_hot @ self.weight
        
        # Reshape to (seq_len, bs, embedding_dim)
        output = embedded.reshape((seq_len, bs, self.embedding_dim))
        
        return output
        ### END YOUR SOLUTION