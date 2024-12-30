from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import zipfile
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

def download_dataset(competition_name: str, download_path: str = "data") -> None:
    """Downloads and extracts Kaggle dataset."""
    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()
    
    print(f"Downloading dataset: {competition_name}...")
    api.competition_download_files(competition_name, download_path)

    zip_path = download_path / f"{competition_name}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        zip_path.unlink()
        print(f"Dataset extracted to: {download_path}")
    else:
        print(f"Error: {zip_path} not found.")

def load_data(competition_name: str, data_path: str = "data") -> tuple[TensorDataset, TensorDataset]:
    data_path = Path(data_path)
    train_path = data_path / "train.csv"
    
    if not train_path.exists():
        download_dataset(competition_name, data_path)
    else:
        print("Dataset found, skipping download...")

    train_dataframe = pd.read_csv(data_path / "train.csv")
    test_dataframe = pd.read_csv(data_path / "test.csv")
    
    # 1. Handle missing values first
    train_dataframe = train_dataframe.fillna(0)  # Fill NA with 0
    test_dataframe = test_dataframe.fillna(0)    # Fill NA with 0
    
    # 2. Separate numeric and categorical columns
    numeric_columns = train_dataframe.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = train_dataframe.select_dtypes(include=['object']).columns
    
    # 3. Process training data
    # Save target variable
    y_train = train_dataframe['SalePrice'].values
    
    # Remove SalePrice from features
    if 'SalePrice' in numeric_columns:
        numeric_columns = numeric_columns.drop('SalePrice')
    
    # Convert categorical columns to dummy variables
    train_categorical = pd.get_dummies(train_dataframe[categorical_columns], drop_first=True)
    test_categorical = pd.get_dummies(test_dataframe[categorical_columns], drop_first=True)
    
    # Get numeric features
    train_numeric = train_dataframe[numeric_columns].astype('float32')  # Convert to float32
    test_numeric = test_dataframe[numeric_columns].astype('float32')    # Convert to float32
    
    # Combine features
    X_train_processed = pd.concat([train_numeric, train_categorical], axis=1)
    X_test_processed = pd.concat([test_numeric, test_categorical], axis=1)
    
    # Ensure columns match
    missing_cols = set(X_train_processed.columns) - set(X_test_processed.columns)
    for col in missing_cols:
        X_test_processed[col] = 0
    X_test_processed = X_test_processed[X_train_processed.columns]
    
    # Convert to float32 to ensure compatibility
    X_train_processed = X_train_processed.astype('float32')
    X_test_processed = X_test_processed.astype('float32')
    
    print("Data processed successfully!")
    print(f"Training data shape: {X_train_processed.shape[0]} samples with {X_train_processed.shape[1]} features")
    print(f"Test data shape: {X_test_processed.shape[0]} samples with {X_test_processed.shape[1]} features")

    # Convert to tensors
    x_train = torch.from_numpy(X_train_processed.values)  # Using from_numpy instead of FloatTensor
    y_train = torch.from_numpy(y_train.astype('float32'))  # Convert target to float32
    x_test = torch.from_numpy(X_test_processed.values)
    
    # Create TensorDatasets
    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, torch.zeros(len(x_test)))
    
    return train_data, test_data


competition_name = "house-prices-advanced-regression-techniques"
data_path = "data"

# Load the data (will download if needed)
train_data, test_data = load_data(competition_name, data_path)

batch_size = 64

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# You can verify the data format:
for x, y in train_dataloader:
    print(f"Shape of X: {x.shape}")
    print(f"Shape of y: {y.shape}")
    break


class HousingPriceModel(nn.Module):
    def __init__(self):
        """Initialize the model for housing price prediction."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Add model architecture here
        pass

    
"""def load_data(competition_name: str, data_path: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_path)
    train_path = data_path / "train.csv"
    
    if not train_path.exists():
        download_dataset(competition_name, data_path)
    else:
        print("Dataset found, skipping download...")

    train_dataframe = pd.read_csv(data_path / "train.csv")
    test_dataframe = pd.read_csv(data_path / "test.csv")
    print("Data loaded successfully!")
    print(f"Training data shape: {train_dataframe.shape[0]} samples with {train_dataframe.shape[1]} features/columns")
    print(f"Test data shape: {test_dataframe.shape[0]} samples with {test_dataframe.shape[1]} features/columns")

    numeric_columns = train_dataframe.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = train_dataframe.select_dtypes(include=['object']).columns

    train_categorical = pd.get_dummies(train_dataframe[categorical_columns], drop_first=True)   #get_dummies converts categorical to 1s and 0s
    test_categorical = pd.get_dummies(test_dataframe[categorical_columns], drop_first=True)

    x_train = torch.FloatTensor(train_dataframe.drop('SalePrice', axis = 1).values)     #Drop output variable and convert to Tensor
    y_train = torch.FloatTensor(train_dataframe['SalePrice'].values)       #Convert training answers to Tensor

    X_test = torch.FloatTensor(test_dataframe.values)      #Convert test data to Tensor with no target column
    
    train_data = TensorDataset(x_train, y_train)    #Combines two Tensors into a dataset
    test_data = TensorDataset(X_test, torch.zeros(len(X_test)))  # Dataset with test data and placeholder labels (no SalePrice values)
    
    return train_data, test_data

"""