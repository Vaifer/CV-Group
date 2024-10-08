import os
from datetime import datetime

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import model
from dataSet import train_dataSet

class Trainer:
    def __init__(self, csv_file, img_dir, model, transform, criterion, optimizer, random_state=8, test_size=0.2, batch_size=32, num_epochs=10):
        # Initialize training parameters
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.model = model
        self.stratify_column = 'stable_height'  # Column used for stratified sampling
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move model to GPU if available
        self.random_state = random_state

        # Pre-processing <- transform
        self.transform = transform

        # Loss function and optimizer
        self.criterion = criterion
        self.optimizer = optimizer

        # Load and partition the dataset
        self.data_frame = pd.read_csv(csv_file)
        self.train_loader, self.val_loader = self.split_dataset()

    def split_dataset(self):
        split = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        # data -> Stratified split
        for train_idx, val_idx in split.split(self.data_frame, self.data_frame[self.stratify_column]):
            train_data = self.data_frame.iloc[train_idx]
            val_data = self.data_frame.iloc[val_idx]

        print(f"train_size: {len(train_data)}, val_size: {len(val_data)}")

        train_dataset = train_dataSet(train_data, self.img_dir, transform)
        val_dateset= train_dataSet(train_data, self.img_dir, transform)
        # data -> DataLoader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dateset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    #
    def calculate_accuracy(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return correct / total

    def generate_classification_report(self, outputs, labels):
        """
        Generate classification report.
        :param outputs: Predicted outputs (already processed through torch.max).
        :param labels: Ground truth labels.
        """
        predicted = outputs.cpu().numpy()  # 预测值已经是 torch.max 的结果
        labels = labels.cpu().numpy()  # 标签直接使用
        report = classification_report(labels, predicted, digits=3, zero_division=0)
        print(report)

    def calculate_confusion_matrix(self, outputs, labels):
        """
        Generate confusion matrix.
        :param outputs: Predicted outputs (already processed through torch.max).
        :param labels: Ground truth labels.
        """
        predicted = outputs.cpu().numpy()  # 预测值已经是 torch.max 的结果
        labels = labels.cpu().numpy()  # 标签直接使用
        cm = confusion_matrix(labels, predicted)
        print(cm)

    def validate(self):
        '''Validation on a validation set'''

        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels, _ in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                labels = labels - 1  # (0,5) <- label(1,6)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

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
                    inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                    labels = labels - 1  # (0,5) <- label(1,6)

                    # Gradient Descent & Backpropagation
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
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
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Save the best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), f"{solution_dir}/best_model.pth")
                print("Saved best model")

            print(f"Current best validation accuracy: {best_val_accuracy:.4f}")

        writer.close()

    def create_log_dir(self):
        '''Create a directory where logs and models are saved'''
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = f'runs/experiment_{current_time}'
        solution_dir = f'trained_models/experiment_{current_time}'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(solution_dir, exist_ok=True)
        return log_dir, solution_dir


if __name__ == "__main__":

    # define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # choose model
    model = model.Stack_GoogleNet()  # Use your modified multi-class classification model

    # define optimizer
    criterion = torch.nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Run
    trainer = Trainer(
        csv_file='./COMP90086_2024_Project_train/train.csv',
        img_dir='./COMP90086_2024_Project_train/train',
        model=model,
        transform=transform,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=8
    )

    trainer.train()
