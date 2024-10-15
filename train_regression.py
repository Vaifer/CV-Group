import os
from datetime import datetime

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from models import Stack_GoogleNet,  Stack_Vit
from dataSet import train_dataSet, test_dataSet  # 假设 BlockStackDataset 是定义的数据集类

class BlockStackTrainer:
    def __init__(self, csv_file, img_dir, model, transform, criterion,optimizer,random_state=8,test_size=0.2, batch_size=32, num_epochs=10):
        self.random_state = random_state
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.model = model
        self.stratify_column = 'stable_height'
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # pre-processing <- transform
        self.transform = transform

        # optimizer
        self.criterion = criterion
        self.optimizer = optimizer

        #  Load and partition the dataset
        self.data_frame = pd.read_csv(csv_file)
        self.train_loader, self.val_loader =self.split_dataset()

    def create_dataloader(self, data_frame, transform, shuffle):
        dataSet=train_dataSet(data_frame,self.img_dir,transform)

        return DataLoader(dataSet,self.batch_size,shuffle)

    def split_dataset(self):

        # StratifiedShuffleSplit
        split = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)

        # Split -> stratify_column
        for train_idx, val_idx in split.split(self.data_frame, self.data_frame[self.stratify_column]):
            train_data = self.data_frame.iloc[train_idx]
            val_data = self.data_frame.iloc[val_idx]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # size -> os
        print(f"train_size: {len(train_data)}, val_size: {len(val_data)}")
        train_loader = self.create_dataloader(train_data, self.transform,True)
        val_loader = self.create_dataloader(val_data, transform, False)
        return train_loader, val_loader
    def calculate_accuracy(self, outputs, labels):
        predicted = torch.round(outputs).clamp(min=1, max=6)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy

    def generate_classification_report(self, outputs, labels):
        predicted = torch.round(outputs).clamp(min=1, max=6).cpu().numpy()
        labels = labels.cpu().numpy()
        report = classification_report(labels, predicted, digits=3, zero_division=0)
        print(report)

    def calculate_confusion_matrix(self, outputs, labels):
        predicted = torch.round(outputs).clamp(min=1, max=6).cpu().numpy()
        labels = labels.cpu().numpy()
        cm = confusion_matrix(labels, predicted)
        print(cm)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels, _ in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                outputs = self.model(inputs)
                outputs = torch.round(outputs).clamp(min=1, max=6)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())

                loss = self.criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

                correct_predictions += (outputs.squeeze() == labels).sum().item()
                total_samples += labels.size(0)

        val_accuracy = correct_predictions / total_samples
        self.generate_classification_report(torch.tensor(all_predictions), torch.tensor(all_labels))
        self.calculate_confusion_matrix(torch.tensor(all_predictions), torch.tensor(all_labels))

        return val_loss / len(self.val_loader), val_accuracy

    def train(self):
        log_dir, solution_dir = self.create_log_dir()
        writer = SummaryWriter(log_dir)

        best_val_accuracy = 0.0

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            running_accuracy = 0.0

            with tqdm(self.train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.num_epochs}")
                for inputs, labels, _ in tepoch:
                    inputs, labels = inputs.to(self.device), labels.to(self.device).float()

                    self.optimizer.zero_grad()
                    raw_outputs = self.model(inputs)
                    outputs = torch.round(raw_outputs).clamp(min=1, max=6)
                    loss = self.criterion(raw_outputs.squeeze(), labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    accuracy = self.calculate_accuracy(outputs, labels)
                    running_accuracy += accuracy

                    writer.add_scalar("Training Loss", loss.item(), epoch * len(self.train_loader) + tepoch.n)
                    writer.add_scalar("Training Accuracy", accuracy, epoch * len(self.train_loader) + tepoch.n)

                    tepoch.set_postfix(loss=running_loss / len(self.train_loader), accuracy=running_accuracy / len(self.train_loader))

            writer.add_scalar('Epoch Loss', running_loss / len(self.train_loader), epoch)
            writer.add_scalar('Epoch Accuracy', running_accuracy / len(self.train_loader), epoch)

            val_loss, val_accuracy = self.validate()
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), f"{solution_dir}/best_model.pth")
                print("保存最佳模型")

            print(f"当前最佳验证准确率: {best_val_accuracy:.4f}")

        writer.close()

    def create_log_dir(self):
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = f'runs/experiment_{current_time}'
        solution_dir = f'trained_models/experiment_{current_time}'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(solution_dir, exist_ok=True)
        return log_dir, solution_dir

if __name__ == "__main__":


    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Unify image size
        transforms.ToTensor(),  # -> tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的标准化参数
    ])

    model = Stack_GoogleNet()  # 假设 BlockStackNet 是一个有4个输出类别的模型
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()  # 假设是回归任务
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = BlockStackTrainer(csv_file='./COMP90086_2024_Project_train/train.csv',
                                img_dir='./COMP90086_2024_Project_train/train',
                                model=model,
                                transform=transform,
                                criterion=criterion,
                                optimizer=optimizer,
                                num_epochs=8)

    trainer.train()
