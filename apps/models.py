import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # Initial layer: ConvBN(3, 16, 7, 4)
        self.conv1 = ndl.nn.Conv(3, 16, 7, stride=4, device=device, dtype=dtype)
        self.bn1 = ndl.nn.BatchNorm2d(16, device=device)
        self.relu1 = ndl.nn.ReLU()
        
        # Layer 2: ConvBN(16, 32, 3, 2)
        self.conv2 = ndl.nn.Conv(16, 32, 3, stride=2, device=device, dtype=dtype)
        self.bn2 = ndl.nn.BatchNorm2d(32, device=device)
        self.relu2 = ndl.nn.ReLU()
        
        # Residual block 1: Two ConvBN(32, 32, 3, 1)
        self.conv3_1 = ndl.nn.Conv(32, 32, 3, stride=1, device=device, dtype=dtype)
        self.bn3_1 = ndl.nn.BatchNorm2d(32, device=device)
        self.relu3_1 = ndl.nn.ReLU()
        
        self.conv3_2 = ndl.nn.Conv(32, 32, 3, stride=1, device=device, dtype=dtype)
        self.bn3_2 = ndl.nn.BatchNorm2d(32, device=device)
        self.relu3_2 = ndl.nn.ReLU()
        
        # Layer 4: ConvBN(32, 64, 3, 2)
        self.conv4 = ndl.nn.Conv(32, 64, 3, stride=2, device=device, dtype=dtype)
        self.bn4 = ndl.nn.BatchNorm2d(64, device=device)
        self.relu4 = ndl.nn.ReLU()
        
        # Layer 5: ConvBN(64, 128, 3, 2)
        self.conv5 = ndl.nn.Conv(64, 128, 3, stride=2, device=device, dtype=dtype)
        self.bn5 = ndl.nn.BatchNorm2d(128, device=device)
        self.relu5 = ndl.nn.ReLU()
        
        # Residual block 2: Two ConvBN(128, 128, 3, 1)
        self.conv6_1 = ndl.nn.Conv(128, 128, 3, stride=1, device=device, dtype=dtype)
        self.bn6_1 = ndl.nn.BatchNorm2d(128, device=device)
        self.relu6_1 = ndl.nn.ReLU()
        
        self.conv6_2 = ndl.nn.Conv(128, 128, 3, stride=1, device=device, dtype=dtype)
        self.bn6_2 = ndl.nn.BatchNorm2d(128, device=device)
        self.relu6_2 = ndl.nn.ReLU()
        
        # Final layers
        self.linear1 = ndl.nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu7 = ndl.nn.ReLU()
        self.linear2 = ndl.nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second convolution
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # First residual block
        identity = x
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)
        x = x + identity  # Residual connection
        
        # Fourth convolution
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        # Fifth convolution
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        # Second residual block
        identity = x
        x = self.conv6_1(x)
        x = self.bn6_1(x)
        x = self.relu6_1(x)
        x = self.conv6_2(x)
        x = self.bn6_2(x)
        x = self.relu6_2(x)
        x = x + identity  # Residual connection
        
        # Flatten: assuming x is now (N, 128, 1, 1) -> (N, 128)
        x = x.reshape((x.shape[0], 128))
        
        # Fully connected layers
        x = self.linear1(x)
        x = self.relu7(x)
        x = self.linear2(x)
        
        return x
        ### END YOUR SOLUTION




class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN, LSTM, or Transformer), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn', 'lstm', or 'transformer'
        num_layers: Number of layers in RNN, LSTM, or Transformer
        seq_len: Sequence length (used for Transformer positional embeddings)
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_model_type = seq_model
        self.device = device
        self.dtype = dtype
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        
        # Sequence model (RNN, LSTM, or Transformer)
        # Sequence model (RNN, LSTM, or Transformer)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers, 
                                  device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers,
                                    device=device, dtype=dtype)
        elif seq_model == 'transformer':
            self.seq_model = nn.Transformer(
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_head=8,
                dim_head=32,
                dropout=0.0,
                causal=True,
                device=device,
                dtype=dtype,
                batch_first=False,
                sequence_len=seq_len
            )
        else:
            raise ValueError(f"Unknown seq_model: {seq_model}")
        
        # Output linear layer
        # For Transformer, input is embedding_size; for RNN/LSTM, input is hidden_size
        if seq_model == 'transformer':
            self.linear = nn.Linear(embedding_size, output_size, device=device, dtype=dtype)
        else:
            self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        
        # Embedding: (seq_len, bs) -> (seq_len, bs, embedding_size)
        embedded = self.embedding(x)
        
        # Sequence model
        seq_output, h_new = self.seq_model(embedded, h)
        
        # Reshape for linear layer
        if self.seq_model_type == 'transformer':
            # seq_output: (seq_len, bs, embedding_size) -> (seq_len*bs, embedding_size)
            seq_output_reshaped = seq_output.reshape((seq_len * bs, self.embedding_size))
        else:
            # seq_output: (seq_len, bs, hidden_size) -> (seq_len*bs, hidden_size)
            seq_output_reshaped = seq_output.reshape((seq_len * bs, self.hidden_size))
        
        # Linear layer: (seq_len*bs, embedding_size/hidden_size) -> (seq_len*bs, output_size)
        output = self.linear(seq_output_reshaped)
        
        return output, h_new
        ### END YOUR SOLUTION



if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
