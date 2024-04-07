import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt

class USGCommonDataset(Dataset):
    def __init__(self, csv_file: str, transform=None, train: bool = True):
        """
        Initializes the USGDatasetMultiClassWithPatientSplit.

        Args:
            csv_file (str): Path to the CSV file containing dataset metadata.
                This CSV file provides information about each sample in the dataset, including paths to image and mask files, as well as polygon labels.
            transform (callable, optional): Optional transform to be applied to samples.
                This parameter allows for applying data augmentation or preprocessing techniques to the images and masks before they are used in training or evaluation.
            train (bool, optional): Indicates whether to use the training set (True) or the test set (False).
                Depending on the value of this parameter, the dataset will return samples from either the training or test portion of the data.
        """
        # Read the CSV file and select relevant columns
        self.data_frame = pd.read_csv(csv_file)[["video_id_x", "frame_cropped_path", "mask_cropped_path", "polygon_label"]]
        
        # Filter data based on the specified type of polygon label
        self.data_frame = self.data_frame[self.data_frame["polygon_label"].isin(["lungslidingpresent", "lungslidingabsent"])]
        
        self.transform = transform

        # Split dataset based on patient_id while maintaining the patient split
        train_df, test_df = train_test_split(self.data_frame, test_size=0.2, random_state=42, stratify=self.data_frame['video_id_x'])
        self.data_frame = train_df if train else test_df

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> list:
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            list: A list containing the image and its corresponding mask.
        """
        # Retrieve image and mask paths
        img_path = self.data_frame.iloc[idx, 1]
        mask_path = self.data_frame.iloc[idx, 2]
        
        # Load image and mask using PIL
        image = Image.open(f"{PATH}/"+img_path)
        mask = Image.open(f"{PATH}/"+mask_path)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return [image, mask]




class USGDatasetMultiClassWithFold(Dataset):
    def __init__(self, data_folder_path: str, csv_file: str, fold: int, transform=None, train: bool = True, tp: int = 0):
        """
        Initializes the DatasetMultyClassWithFold.

        Args:
            csv_file (str): Path to the CSV file containing dataset metadata.
                This CSV file provides information about each sample in the dataset, including paths to image and mask files, as well as polygon labels.
            fold (int): Fold number for cross-validation.
                The dataset may be divided into multiple folds for cross-validation purposes. This parameter specifies which fold to use for either training or validation.
            transform (callable, optional): Optional transform to be applied to samples.
                This parameter allows for applying data augmentation or preprocessing techniques to the images and masks before they are used in training or evaluation.
            train (bool, optional): Indicates whether to use the training set (True) or the validation set (False).
                Depending on the value of this parameter, the dataset will select samples from either the training or validation portion of the data.
            tp (int, optional): Type of polygon label to filter the dataset.
                This parameter allows filtering the dataset based on the type of polygon label associated with each sample. 
                It can be set to 0 for lung sliding, 1 for aline, or 2 for bline, selecting only samples with the corresponding label type.
        """
        # Read the CSV file and select relevant columns
        self.data_folder_path = data_folder_path
        self.data_frame = pd.read_csv(csv_file)[["video_id_x", "frame_cropped_path", "mask_cropped_path", "polygon_label"]]
        
        # Filter data based on the specified type of polygon label
        if tp == 0:
            self.data_frame = self.data_frame[self.data_frame["polygon_label"].isin(["lungslidingpresent", "lungslidingabsent"])]
        elif tp == 1:
            self.data_frame = self.data_frame[self.data_frame["polygon_label"] == "aline"]
        elif tp == 2:
            self.data_frame = self.data_frame[self.data_frame["polygon_label"] == "bline"]
        
        self.transform = transform


        # Cross validation - explanation:
        # - Initialize a KFold object with 5 folds for cross-validation. Data is shuffled, and a random seed is set to 42 for reproducibility.
        # - Extract patient IDs from the DataFrame to be used for splitting the dataset.
        # - Iterate through each fold, using enumerate to keep track of the fold number and corresponding training/validation indices.
        # - When the desired fold number (fold) is reached:
        #   - If train is True, update the dataset with samples for training using the indices from train_index.
        #   - If train is False, update the dataset with samples for validation using the indices from val_index.
        # - Exit the loop once the desired fold is processed to avoid unnecessary iterations.

        # Create folds for cross-validation based on patient_id
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Initialize KFold object with 5 folds, shuffling the data and setting a random seed for reproducibility
        patient_ids = self.data_frame['video_id_x'].values  # Get the patient IDs from the DataFrame

        # Iterate through the folds and select the desired fold for either training or validation
        for fold_num, (train_index, val_index) in enumerate(kf.split(X=range(len(patient_ids)))):
            # 'fold_num' keeps track of the current fold number
            # 'train_index' contains the indices for the training set, and 'val_index' contains the indices for the validation set
            
            # Check if the current fold matches the desired fold number
            if fold_num == fold:
                # If it matches, update the dataset based on whether it's for training or validation
                if train:
                    # If it's for training, select the samples corresponding to the training indices
                    self.data_frame = self.data_frame.iloc[train_index]
                else:
                    # If it's for validation, select the samples corresponding to the validation indices
                    self.data_frame = self.data_frame.iloc[val_index]
                
                # Exit the loop once the desired fold is processed
                break

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> list:
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            list: A list containing the image and its corresponding mask.
        """
        # Retrieve image and mask paths
        img_path = self.data_frame.iloc[idx, 1]
        mask_path = self.data_frame.iloc[idx, 2]
        
        # Load image and mask using PIL
        image = Image.open(f"{self.data_folder_path}/"+img_path)
        mask = Image.open(f"{self.data_folder_path}/"+mask_path)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return [image, mask]
