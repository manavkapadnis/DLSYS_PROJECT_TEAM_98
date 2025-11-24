import os
import numpy as np
from needle import backend_ndarray as nd
from needle import Tensor


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            # Add new word
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION


class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        ids = []
        
        with open(path, 'r') as f:
            lines_read = 0
            for line in f:
                # Stop if we've read max_lines
                if max_lines is not None and lines_read >= max_lines:
                    break
                
                # Split line into words
                words = line.split()
                
                # Add each word to dictionary and collect IDs
                for word in words:
                    ids.append(self.dictionary.add_word(word))
                
                # Add end-of-sentence token
                ids.append(self.dictionary.add_word('<eos>'))
                
                lines_read += 1
        
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    # Calculate number of batches (trim remainder)
    nbatch = len(data) // batch_size
    
    # Trim data to fit evenly into batches
    data = data[:nbatch * batch_size]
    
    # Reshape into (batch_size, nbatch) then transpose to (nbatch, batch_size)
    # This arranges sequential data into columns
    data = np.array(data).reshape(batch_size, nbatch).T
    
    return data
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    # Determine the actual sequence length (might be less than bptt at the end)
    seq_len = min(bptt, len(batches) - 1 - i)
    
    # Extract data: rows from i to i+seq_len
    data = batches[i:i+seq_len, :]
    
    # Extract targets: rows from i+1 to i+seq_len+1 (shifted by 1)
    target = batches[i+1:i+seq_len+1, :]
    
    # Flatten target to 1D
    target = target.flatten()
    
    # Convert to Tensors
    data = Tensor(data, device=device, dtype=dtype)
    target = Tensor(target, device=device, dtype=dtype)
    
    return data, target
    ### END YOUR SOLUTION
