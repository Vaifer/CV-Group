import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataSet import test_dataSet  # Assuming test_dataSet is the class to load test data
from model import Stack_GoogleNet


class StackPredictor:
    def __init__(self, model, model_path, test_csv, img_dir, batch_size=32):

        # initialize
        self.model = model
        self.model_path = model_path
        self.test_csv = test_csv
        self.img_dir = img_dir
        self.batch_size = batch_size

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Load the trained model weights
        self.load_model()

        # Define the image transformations (same as during training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """Load the trained model weights from the specified path."""
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()  # Set the model to evaluation mode

    def predict(self):
        """Predict the labels for the test dataset."""
        all_predictions = []
        all_image_ids = []

        # Load test data
        test_data_frame = pd.read_csv(self.test_csv)
        test_dataset = test_dataSet(test_data_frame, self.img_dir, self.transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Perform prediction
        with torch.no_grad():
            for images, image_ids in tqdm(test_loader, desc="Predicting"):
                images = images.to(self.device)

                # Get model predictions
                outputs = self.model(images)

                # Use torch.max to get the predicted class (index) for multi-class classification
                _, predicted = torch.max(outputs, 1)

                # Since our labels range from 1 to 6, we adjust the predicted index by adding 1
                predicted = predicted + 1

                # Convert predictions to int and store them
                all_predictions.extend(predicted.cpu().numpy())
                all_image_ids.extend(image_ids.numpy())  # Convert image_ids to numpy and extend

        # Create DataFrame for predictions
        prediction_df = pd.DataFrame({
            'id': all_image_ids,
            'stable_height': all_predictions
        })

        return prediction_df

    def save_predictions(self, output_csv):
        """Save the predictions to a CSV file."""
        prediction_df = self.predict()
        prediction_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    # Define the model (assuming it's the same architecture used during training)
    model = Stack_GoogleNet()

    # Paths to required files
    model_path = 'trained_models/experiment_2024-10-08_15-49-29/best_model.pth'  # Path to the saved model
    test_csv = 'COMP90086_2024_Project_test/test.csv'  # Path to the CSV file containing test data
    img_dir = 'COMP90086_2024_Project_test/test'  # Path to the directory containing images

    # Create predictor instance
    predictor = StackPredictor(
        model=model,
        model_path=model_path,
        test_csv=test_csv,
        img_dir=img_dir,
        batch_size=32  # Adjust batch size as needed
    )

    # Make predictions
    prediction_df = predictor.predict()

    # Save predictions to a CSV file
    predictor.save_predictions('predictions.csv')
