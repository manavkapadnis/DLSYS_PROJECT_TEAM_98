import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        # Reset the iteration state for new epoch
        self.current_batch = 0
        
        # Create ordering for this epoch
        if self.shuffle:
            # Shuffle indices and split into batches
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices, 
                                        range(self.batch_size, len(self.dataset), self.batch_size))
        # Note: If shuffle=False, ordering is already pre-computed in __init__
        
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        # Check if we've exhausted all batches
        if self.current_batch >= len(self.ordering):
            raise StopIteration
        
        # Get indices for current batch
        batch_indices = self.ordering[self.current_batch]
        
        # Build separate lists for samples and labels
        batch_samples = []
        batch_labels = []
        
        # Fetch data samples for this batch
        for i in batch_indices:
            result = self.dataset[i]
            
            # Check if dataset returns (data, label) tuple or just data
            if isinstance(result, tuple) and len(result) == 2:
                # MNIST case: returns (image, label)
                sample, label = result
            else:
                # NDArrayDataset case: returns single value or 1-tuple
                if isinstance(result, tuple):
                    sample = result[0]
                else:
                    sample = result
                # For NDArrayDataset, use sample as both data and label
                label = sample
            
            batch_samples.append(sample)
            batch_labels.append(label)
        
        # Stack into batches using numpy
        batch_images = np.stack(batch_samples, axis=0)  # Shape: (batch_size, ...)
        batch_labels = np.stack(batch_labels, axis=0)   # Shape: (batch_size, ...)
        
        # Convert to Tensors
        from ..autograd import Tensor
        batch_images = Tensor(batch_images)
        batch_labels = Tensor(batch_labels)
        
        # Move to next batch
        self.current_batch += 1
        
        return (batch_images, batch_labels)
        ### END YOUR SOLUTION