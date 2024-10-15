
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

    def baseline_trainer(self,transform, lr=0.001, num_epochs=40):
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
        model = models.Stack_Resnet(num_classes=2)  # Use your modified multi-class classification model

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
        model = models.Stack_Inception(num_classes=3)  # Use your modified multi-class classification model

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

    def baseline_trainer_divideby_total_height(self,transform,height, lr=0.001, num_epochs=16):
        '''
        Baseline for stable_height classification
        :param lr:
        :param num_epochs:
        :return:
        '''
        # choose model
        model = models.Stack_GoogleNet(num_classes=height)  # Use your modified multi-class classification model

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
            batch_size=32,
            stratify_column="stable_height",
            clean_column="total_height",
            clean_column_value=height,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs
        )

    def baseline_trainer_part(self,transform, lr=0.001, num_epochs=40):
        '''
        Baseline for stable_height classification
        :param lr:
        :param num_epochs:
        :return:
        '''
        # choose model
        model = models.Stack_Inception(num_classes=6)  # Use your modified multi-class classification model

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
            batch_size=32,
            stratify_column="stable_height",
            clean_column="instability_type",
            clean_column_value=2,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs
        )
    def single_trainer(self,transform, lr=0.001, num_epochs=40):
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
            column_set=4,
            transform= transform,
            batch_size=32,
            stratify_column="stable_height",
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

    transform1 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set=train_set(train_csv_dir,train_img_dir,transform)
    #choose trainer
    # baseline:用于尝试各种model
    trainer=train_set.baseline_trainer(transform)

    # instabilityType :训练 分类 不稳定类型
    # trainer=train_set.instabilityType_trainer(transform)

    # height_trainer :训练 分类 totol height
    # trainer = train_set.height_trainer(transform)

    #non_planar： 训练 分类是否非平面
    # trainer = train_set.shapeset_trainer(transform)

    # 分类训练
    # trainer=train_set.baseline_trainer_part(transform)
    # trainer=train_set.single_trainer(transform)

    trainer.train()