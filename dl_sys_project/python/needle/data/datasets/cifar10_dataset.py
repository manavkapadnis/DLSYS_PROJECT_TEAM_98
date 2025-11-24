import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        super().__init__(transforms)
        
        # Load the appropriate data files
        if train:
            # Load all 5 training batches
            data_batches = []
            labels_batches = []
            
            for i in range(1, 6):
                batch_file = os.path.join(base_folder, f'data_batch_{i}')
                with open(batch_file, 'rb') as f:
                    batch_dict = pickle.load(f, encoding='bytes')
                    data_batches.append(batch_dict[b'data'])
                    labels_batches.append(batch_dict[b'labels'])
            
            # Concatenate all training batches
            self.X = np.concatenate(data_batches, axis=0)
            self.y = np.concatenate(labels_batches, axis=0)
        else:
            # Load test batch
            test_file = os.path.join(base_folder, 'test_batch')
            with open(test_file, 'rb') as f:
                test_dict = pickle.load(f, encoding='bytes')
                self.X = test_dict[b'data']
                self.y = np.array(test_dict[b'labels'])
        
        # Reshape data from (N, 3072) to (N, 3, 32, 32)
        # CIFAR-10 data is stored as (N, 3072) where 3072 = 3 * 32 * 32
        # The first 1024 values are red channel, next 1024 are green, last 1024 are blue
        self.X = self.X.reshape(-1, 3, 32, 32)
        
        # Normalize to [0, 1] range
        self.X = self.X.astype(np.float32) / 255.0
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # Get image and label at index
        img = self.X[index]
        label = self.y[index]
        
        # Apply transforms if they exist
        # Note: transforms expect HWC format, so we need to transpose
        if self.transforms is not None:
            # Transpose from (3, 32, 32) to (32, 32, 3) for transforms
            img = img.transpose(1, 2, 0)
            # Apply transforms
            img = self.apply_transforms(img)
            # Transpose back to (3, 32, 32)
            img = img.transpose(2, 0, 1)
        
        return (img, label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return self.X.shape[0]
        ### END YOUR SOLUTION
