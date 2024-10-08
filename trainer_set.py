
import torch
import torch.optim as optim
from torchvision import transforms

import models
from train import Trainer

class train_set:

    def __init__(self,csv_file,img_dir):
        self.csv_file = csv_file
        self.img_dir = img_dir

    def baseline_trainer(self,lr=0.001,num_epochs=8):
        '''
        Baseline for stable_height classification
        :param lr:
        :param num_epochs:
        :return:
        '''
        # choose model
        model = models.Stack_GoogleNet()  # Use your modified multi-class classification model

        # define optimizer
        criterion = torch.nn.CrossEntropyLoss()  # For multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=lr)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # define Trainer
        return Trainer(
            self.csv_file,
            self.img_dir,
            model=model,
            transform= transform,
            stratify_column="stable_height",
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs
        )

    def instabilityType_trainer(self,lr=0.001,num_epochs=8):
        '''
        Baseline for stable_height classification
        :param lr:
        :param num_epochs:
        :return:
        '''
        # choose model
        model = models.Stack_Inception(num_classes=3)  # Use your modified multi-class classification model

        # define optimizer
        criterion = torch.nn.CrossEntropyLoss()  # For multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=lr)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # define Trainer
        return Trainer(
            self.csv_file,
            self.img_dir,
            model=model,
            transform= transform,
            stratify_column="instability_type",
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs
        )
    def height_trainer(self,lr=0.001,num_epochs=8):
        '''
        Baseline for stable_height classification
        :param lr:
        :param num_epochs:
        :return:
        '''
        # choose model
        model = models.Stack_GoogleNet()  # Use your modified multi-class classification model

        # define optimizer
        criterion = torch.nn.CrossEntropyLoss()  # For multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=lr)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # define Trainer
        return Trainer(
            self.csv_file,
            self.img_dir,
            model=model,
            transform= transform,
            stratify_column="total_height",
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs
        )

if __name__ == "__main__":
    train_csv_dir = './COMP90086_2024_Project_train/train.csv'
    train_img_dir = './COMP90086_2024_Project_train/train'

    train_set=train_set(train_csv_dir,train_img_dir)
    #choose trainer
    # baseline:用于尝试各种model
    trainer=train_set.baseline_trainer()

    # instabilityType :训练 分类 不稳定类型
    # trainer=train_set.instabilityType_trainer()

    # height_trainer :训练 分类 totol height
    # trainer = train_set.height_trainer()


    trainer.train()