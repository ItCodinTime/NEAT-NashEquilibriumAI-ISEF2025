#!/usr/bin/env python3
"""
CSV Data Loader for Iris Dataset
Provides utilities for loading and preprocessing CSV data files for NEAT training
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IrisDataset(Dataset):
    """Custom PyTorch Dataset for Iris data"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        """
        Initialize the dataset
        
        Args:
            features: Feature array (n_samples, n_features)
            labels: Label array (n_samples,)
            transform: Optional transform to be applied on features
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label

class CSVDataLoader:
    """CSV Data Loader with preprocessing capabilities"""
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.class_names = []
        
    def load_iris_from_csv(self, filename: str = 'iris.data') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Iris dataset from CSV file
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Tuple of (features, labels)
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Define column names for Iris dataset
        column_names = [
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'
        ]
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath, names=column_names, header=None)
            
            # Remove any empty rows
            df = df.dropna()
            
            # Separate features and labels
            features = df.iloc[:, :-1].values
            labels = df.iloc[:, -1].values
            
            # Store feature names and class names
            self.feature_names = column_names[:-1]
            self.class_names = np.unique(labels)
            
            logger.info(f"Loaded {len(df)} samples from {filename}")
            logger.info(f"Features: {self.feature_names}")
            logger.info(f"Classes: {self.class_names}")
            
            return features, labels
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise
    
    def preprocess_data(self, features: np.ndarray, labels: np.ndarray, 
                       normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data (normalization, encoding)
        
        Args:
            features: Raw feature data
            labels: Raw label data
            normalize: Whether to normalize features
            
        Returns:
            Tuple of (processed_features, processed_labels)
        """
        processed_features = features.copy()
        processed_labels = labels.copy()
        
        # Normalize features if requested
        if normalize:
            processed_features = self.scaler.fit_transform(processed_features)
            logger.info("Features normalized using StandardScaler")
        
        # Encode labels
        processed_labels = self.label_encoder.fit_transform(processed_labels)
        logger.info(f"Labels encoded: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return processed_features, processed_labels
    
    def create_data_splits(self, features: np.ndarray, labels: np.ndarray,
                          train_ratio: float = 0.7, val_ratio: float = 0.15,
                          test_ratio: float = 0.15, random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train, validation, and test sets
        
        Args:
            features: Feature array
            labels: Label array
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train, val, test splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, 
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: val vs test
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_test_ratio),
            random_state=random_state,
            stratify=y_temp
        )
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def create_dataloaders(self, data_splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
                          batch_size: int = 32, shuffle_train: bool = True) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders from data splits
        
        Args:
            data_splits: Dictionary with train, val, test data
            batch_size: Batch size for DataLoaders
            shuffle_train: Whether to shuffle training data
            
        Returns:
            Dictionary of DataLoaders
        """
        dataloaders = {}
        
        for split_name, (features, labels) in data_splits.items():
            dataset = IrisDataset(features, labels)
            shuffle = shuffle_train if split_name == 'train' else False
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            dataloaders[split_name] = dataloader
            logger.info(f"Created {split_name} DataLoader with {len(dataset)} samples")
        
        return dataloaders
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data
        
        Returns:
            Dictionary with data information
        """
        return {
            'feature_names': self.feature_names,
            'class_names': self.class_names.tolist() if hasattr(self.class_names, 'tolist') else self.class_names,
            'num_features': len(self.feature_names),
            'num_classes': len(self.class_names),
            'scaler_params': {
                'mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'std': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
            },
            'label_mapping': dict(zip(self.label_encoder.classes_, 
                                    range(len(self.label_encoder.classes_)))) if hasattr(self.label_encoder, 'classes_') else None
        }
    
    def load_and_prepare_iris(self, batch_size: int = 32, 
                             normalize: bool = True,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
        """
        Complete pipeline to load and prepare Iris dataset
        
        Args:
            batch_size: Batch size for DataLoaders
            normalize: Whether to normalize features
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Tuple of (dataloaders, data_info)
        """
        logger.info("Starting Iris data loading and preprocessing pipeline...")
        
        # Load raw data
        features, labels = self.load_iris_from_csv()
        
        # Preprocess data
        features, labels = self.preprocess_data(features, labels, normalize=normalize)
        
        # Create data splits
        data_splits = self.create_data_splits(features, labels, train_ratio, val_ratio, test_ratio)
        
        # Create DataLoaders
        dataloaders = self.create_dataloaders(data_splits, batch_size=batch_size)
        
        # Get data information
        data_info = self.get_data_info()
        
        logger.info("Data loading and preprocessing completed successfully!")
        
        return dataloaders, data_info

def load_custom_csv(filepath: str, feature_cols: list, label_col: str,
                   batch_size: int = 32, normalize: bool = True) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    """
    Load custom CSV data with specified columns
    
    Args:
        filepath: Path to CSV file
        feature_cols: List of feature column names
        label_col: Name of label column
        batch_size: Batch size for DataLoaders
        normalize: Whether to normalize features
        
    Returns:
        Tuple of (dataloaders, data_info)
    """
    try:
        df = pd.read_csv(filepath)
        
        # Extract features and labels
        features = df[feature_cols].values
        labels = df[label_col].values
        
        # Create data loader instance
        loader = CSVDataLoader()
        loader.feature_names = feature_cols
        loader.class_names = np.unique(labels)
        
        # Preprocess
        features, labels = loader.preprocess_data(features, labels, normalize=normalize)
        
        # Create splits
        data_splits = loader.create_data_splits(features, labels)
        
        # Create dataloaders
        dataloaders = loader.create_dataloaders(data_splits, batch_size=batch_size)
        
        # Get info
        data_info = loader.get_data_info()
        
        logger.info(f"Successfully loaded custom CSV data from {filepath}")
        
        return dataloaders, data_info
        
    except Exception as e:
        logger.error(f"Error loading custom CSV data: {str(e)}")
        raise

def demo_data_loading():
    """Demonstration of data loading functionality"""
    print("=" * 50)
    print("IRIS DATA LOADER DEMONSTRATION")
    print("=" * 50)
    
    # Initialize data loader
    loader = CSVDataLoader()
    
    try:
        # Load and prepare data
        dataloaders, data_info = loader.load_and_prepare_iris(batch_size=16)
        
        # Display data information
        print("\nData Information:")
        for key, value in data_info.items():
            print(f"  {key}: {value}")
        
        # Display DataLoader information
        print("\nDataLoader Information:")
        for split_name, dataloader in dataloaders.items():
            print(f"  {split_name}: {len(dataloader.dataset)} samples, {len(dataloader)} batches")
        
        # Show sample batch
        train_loader = dataloaders['train']
        sample_features, sample_labels = next(iter(train_loader))
        print(f"\nSample batch shape: Features {sample_features.shape}, Labels {sample_labels.shape}")
        print(f"Sample features (first 3):\n{sample_features[:3]}")
        print(f"Sample labels (first 10): {sample_labels[:10]}")
        
        return dataloaders, data_info
        
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Run demonstration
    demo_data_loading()
