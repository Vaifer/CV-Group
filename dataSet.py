import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class train_dataSet(Dataset):

    """
    Load the dataset
    """
    def __init__(self, data_frame, img_dir, column_set=0, transform=None):

        self.data_frame = data_frame
        self.img_dir = img_dir
        self.transform = transform
        self.column_set=column_set

        # check directory and access
        if not os.path.exists(self.img_dir):
            raise ValueError(f"directory does not exist: {self.img_dir}")
        if not os.access(self.img_dir, os.R_OK):
            raise PermissionError(f"No access: {self.img_dir}")

    def __len__(self):
        # sample size
        return len(self.data_frame)

    def __getitem__(self, idx):
        # load image
        img_name = os.path.join(self.img_dir, str(self.data_frame.iloc[idx, 0]) + ".jpg")
        image = Image.open(img_name).convert('RGB')

        # column
        stable_height = self.data_frame.iloc[idx, -1]
        # whether the stack contains (1) only cubes or (2) multiple shapes
        shapeset = self.data_frame.iloc[idx, 1]
        # (1) easy or (2) hard
        type = self.data_frame.iloc[idx, 2]
        # the number of objects in the stack
        total_height = self.data_frame.iloc[idx, 3]
        # (0)stable, (1) unstable due to unsupported centre of mass, (2) unstable due to stacking on non-planar surface
        instability_type = self.data_frame.iloc[idx, 4]
        # (1) low or (2) high
        cam_angle = self.data_frame.iloc[idx, 5]

        # Create new column
        #if stable
        # instability_type = 0 if instability_type == 0 else 1
        # if non-planar surface
        non_planar = 1 if instability_type == 2 else 0

        # preprocessing <- transform
        if self.transform:
            image = self.transform(image)


        # Provide column data as required
        if self.column_set ==1:# ->total_height
            return image, total_height , instability_type

        elif self.column_set ==2: # ->instability_type
            return image, instability_type,stable_height
        elif self.column_set == -3: #-> non_planar/stable
            return image, shapeset, stable_height
        elif self.column_set == 4:
            stable_height = 1 if instability_type == 1 else 0
            return image,stable_height,instability_type
        else:#default: ->stable_height
            return image, stable_height , instability_type

class test_dataSet(Dataset):
    def __init__(self, data_frame, img_dir, transform=None):
        """
        Load the data to be predicted
        """
        # csv -> data_frame
        self.data_frame = data_frame
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """sample amount"""
        return len(self.data_frame)

    def __getitem__(self, idx):

        # load image
        img_name = os.path.join(self.img_dir, str(self.data_frame.iloc[idx, 0]) + ".jpg")
        image = Image.open(img_name).convert('RGB')

        # preprocessing <- transform
        if self.transform:
            image = self.transform(image)

        # get image id
        img_id = self.data_frame.iloc[idx, 0]
        return image, img_id


