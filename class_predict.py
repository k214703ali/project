import torch
import joblib
import torch.nn.functional as F
import skimage.io
import torchvision
import torchxrayvision as xrv

import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPClassifier, self).__init__()

        # Increase the size of each layer
        self.fc1 = nn.Linear(input_dim, 1024)  # First hidden layer with more neurons
        self.fc2 = nn.Linear(1024, 512)  # Second hidden layer with more neurons
        self.fc3 = nn.Linear(512, 256)  # Third hidden layer
        self.fc4 = nn.Linear(256, 128)  # Fourth hidden layer
        self.fc5 = nn.Linear(128, 64)  # Fifth hidden layer
        self.fc6 = nn.Linear(64, output_dim)  # Output layer with 18 neurons (for 18 classes)

        # Adding Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

        # Dropout layers
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply Softmax along the output dimension

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))  # First layer + ReLU + BatchNorm
        x = self.dropout(x)  # Dropout for regularization
        x = self.relu(self.bn2(self.fc2(x)))  # Second layer + ReLU + BatchNorm
        x = self.dropout(x)  # Dropout
        x = self.relu(self.bn3(self.fc3(x)))  # Third layer + ReLU + BatchNorm
        x = self.dropout(x)  # Dropout
        x = self.relu(self.bn4(self.fc4(x)))  # Fourth layer + ReLU + BatchNorm
        x = self.dropout(x)  # Dropout
        x = self.relu(self.bn5(self.fc5(x)))  # Fifth layer + ReLU + BatchNorm
        x = self.dropout(x)  # Dropout
        x = self.fc6(x)  # Output layer
        x = self.softmax(x)  # Apply Softmax to get probabilities
        return x


# Load the trained model
model_save_path = "combined_model.pth"
loaded_model = MLPClassifier(input_dim=4608, output_dim=18)  # Adjust input_dim/output_dim accordingly
loaded_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
loaded_model.eval()  # Set the model to evaluation mode
print(f"Model loaded from {model_save_path}")

# Load the label encoder
label_encoder_save_path = "label_encoder.pkl"
loaded_label_encoder = joblib.load(label_encoder_save_path)
print(f"Label encoder loaded from {label_encoder_save_path}")

# Updated load_image function
def load_image(image_path, device):
    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)  # Normalize the image
    if len(img.shape) > 2:  # Convert to grayscale if necessary
        img = img.mean(2)[None, ...]
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = transform(img)
    img = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension
    return img.to(device)

# Preprocess the image
def preprocess_input(image_path, device):
    # Load and preprocess the image
    img_tensor = load_image(image_path, device)
    
    # Extract features using the autoencoder
    z = ae.encode(img_tensor)  # Get the encoded features from the autoencoder
    flattened_features = z.view(z.size(0), -1)  # Flatten the feature map to 1D
    return flattened_features.squeeze().detach().cpu().numpy()  # Remove batch dimension and convert to numpy

# Prediction function
def predict(model, image_path, label_encoder, device):
    # Preprocess the input image and extract features
    input_data = preprocess_input(image_path, device)
    
    # Convert the features to a tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor.to(device))  # Forward pass
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()  # Get the predicted class index
        
    # Decode the label
    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
    return predicted_label, probabilities.cpu().numpy()  # Return the predicted label and the probabilities

# Example usage:
device = torch.device("cpu")  # Change to "cuda" if using GPU
image_path = "3.jpg"  # Replace with the actual image path

ae = xrv.autoencoders.ResNetAE(weights="101-elastic").to(device)

predicted_label, probabilities = predict(loaded_model, image_path, loaded_label_encoder, device)

print(f"Predicted label: {predicted_label}")
