
import torch
import torch.optim as optim
from torchvision import transforms

import models
from train import Trainer

class train_set:

    def __init__(self,csv_file,img_dir,transform):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.transform=transform

    def baseline_trainer(self,transform, lr=0.001, num_epochs=16):
        '''
        Baseline for stable_height classification
        :param lr:
        :param num_epochs:
        :return:
        '''
        # choose model
        model = models.Stack_Inception()  # Use your modified multi-class classification model

        # define optimizer
        criterion = torch.nn.CrossEntropyLoss()  # For multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # define Trainer
        return Trainer(
            self.csv_file,
            self.img_dir,
            model=model,
            column_set=0,
            transform= transform,
            stratify_column="stable_height",
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs
        )

    def instabilityType_trainer(self,transform,lr=0.001,num_epochs=8):
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
        # define Trainer
        return Trainer(
            self.csv_file,
            self.img_dir,
            model=model,
            column_set=2,
            transform= transform,
            stratify_column="instability_type",
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs
        )
    def height_trainer(self,transform,lr=0.001,num_epochs=8):
        '''
        Baseline for stable_height classification
        :param lr:
        :param num_epochs:
        :return:
        '''
        # choose model
        model = models.Stack_Inception()  # Use your modified multi-class classification model

        # define optimizer
        criterion = torch.nn.CrossEntropyLoss()  # For multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # define Trainer
        return Trainer(
            self.csv_file,
            self.img_dir,
            model=model,
            column_set=1,
            transform= transform,
            stratify_column="total_height",
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs
        )
    def shapeset_trainer(self,transform,lr=0.001,num_epochs=8):
        '''
        Baseline for stable_height classification
        :param lr:
        :param num_epochs:
        :return:
        '''
        # choose model
        model = models.Stack_Inception(num_classes=2)  # Use your modified multi-class classification model

        # define optimizer
        criterion = torch.nn.CrossEntropyLoss()  # For multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # define Trainer
        return Trainer(
            self.csv_file,
            self.img_dir,
            model=model,
            column_set= -3,
            transform= transform,
            stratify_column="shapeset",
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs
        )
if __name__ == "__main__":
    train_csv_dir = './COMP90086_2024_Project_train/train.csv'
    train_img_dir = './COMP90086_2024_Project_train/train'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set=train_set(train_csv_dir,train_img_dir,transform)
    #choose trainer
    # baseline:用于尝试各种model
    # trainer=train_set.baseline_trainer(transform)

    # instabilityType :训练 分类 不稳定类型
    # trainer=train_set.instabilityType_trainer(transform)

    # height_trainer :训练 分类 totol height
    # trainer = train_set.height_trainer(transform)

    #non_planar： 训练 分类是否非平面
    trainer = train_set.shapeset_trainer(transform)


    trainer.train()