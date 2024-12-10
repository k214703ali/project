import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
from fpdf import FPDF
from io import BytesIO
import base64
import markdown
from bs4 import BeautifulSoup
import torch
from torchvision import transforms
import skimage.io
import torchvision
import torchxrayvision as xrv
import torch.nn as nn
import torch.nn.functional as F
import joblib
from modules.tokenizers import Tokenizer
from models.models import BaseCMNModel


# -----------------------------#
#        Classification     #
# -----------------------------#

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPClassifier, self).__init__()

        # Increase the size of each layer
        # First hidden layer with more neurons
        self.fc1 = nn.Linear(input_dim, 1024)
        # Second hidden layer with more neurons
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)  # Third hidden layer
        self.fc4 = nn.Linear(256, 128)  # Fourth hidden layer
        self.fc5 = nn.Linear(128, 64)  # Fifth hidden layer
        # Output layer with 18 neurons (for 18 classes)
        self.fc6 = nn.Linear(64, output_dim)

        # Adding Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

        # Dropout layers
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability

        self.relu = nn.ReLU()
        # Apply Softmax along the output dimension
        self.softmax = nn.Softmax(dim=1)

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


# -----------------------------#
#        Report Generation     #
# -----------------------------#
class Args:
    def __init__(self):
        self.image_dir = 'images/'  # updated default
        self.ann_path = 'annotation.json'  # updated default
        self.dataset_name = 'iu_xray'  # updated default
        self.max_seq_length = 60  # updated default
        self.threshold = 3  # updated default
        self.num_workers = 2  # default value (unchanged)
        self.batch_size = 8  # updated default
        self.visual_extractor = 'resnet101'  # default value (unchanged)
        self.visual_extractor_pretrained = True  # default value (unchanged)
        self.d_model = 512  # default value (unchanged)
        self.d_ff = 512  # default value (unchanged)
        self.d_vf = 2048  # default value (unchanged)
        self.num_heads = 8  # default value (unchanged)
        self.num_layers = 3  # updated default
        self.dropout = 0.1  # default value (unchanged)
        self.logit_layers = 1  # default value (unchanged)
        self.bos_idx = 0  # default value (unchanged)
        self.eos_idx = 0  # default value (unchanged)
        self.pad_idx = 0  # default value (unchanged)
        self.use_bn = 0  # default value (unchanged)
        self.drop_prob_lm = 0.5  # default value (unchanged)
        self.topk = 32  # updated default
        self.cmm_size = 2048  # updated default
        self.cmm_dim = 512  # updated default
        self.sample_method = 'beam_search'  # default value (unchanged)
        self.beam_size = 3  # updated default
        self.temperature = 1.0  # default value (unchanged)
        self.sample_n = 1  # default value (unchanged)
        self.group_size = 1  # default value (unchanged)
        self.output_logsoftmax = 1  # default value (unchanged)
        self.decoding_constraint = 0  # default value (unchanged)
        self.block_trigrams = 1  # default value (unchanged)
        self.n_gpu = 1  # default value (unchanged)
        self.epochs = 100  # updated default
        self.save_dir = '/content/drive/MyDrive/results/iu_xray'  # updated default
        self.resume = '/content/drive/MyDrive/results/iu_xray'
        self.record_dir = 'records/'  # default value (unchanged)
        self.log_period = 50  # updated default
        self.save_period = 1  # default value (unchanged)
        self.monitor_mode = 'max'  # default value (unchanged)
        self.monitor_metric = 'BLEU_4'  # default value (unchanged)
        self.early_stop = 50  # default value (unchanged)
        self.optim = 'Adam'  # default value (unchanged)
        self.lr_ve = 1e-4  # updated default
        self.lr_ed = 5e-4  # updated default
        self.step_size = 10  # updated default
        self.gamma = 0.8  # updated default
        self.adam_betas = (0.9, 0.98)  # default value (unchanged)
        self.adam_eps = 1e-9  # default value (unchanged)
        self.amsgrad = True  # default value (unchanged)
        self.noamopt_warmup = 5000  # default value (unchanged)
        self.noamopt_factor = 1  # default value (unchanged)
        self.lr_scheduler = 'StepLR'  # default value (unchanged)
        self.seed = 7580  # updated default
        self.resume = None  # default value (unchanged)
        self.weight_decay = 5e-5


# -----------------------------#
#        Custom CSS Styling     #
# -----------------------------#
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    .stButton>button {
        background-color: #23272F; color: white; border: none;
        padding: 10px 20px; font-size: 16px; border-radius: 5px; cursor: pointer;
    }
    .stButton>button:hover { background-color: #006666; }
    .main-title { font-family: 'Product Sans', sans-serif; font-size: 36px; 
                  color: #23272F; text-align: left; margin-top: -70px; }
    .custom-text { color: #23272F;font-size:16px;font-family: 'Product Sans', sans-serif;margin-bottom:-20px }
    .prediction { font-size: 24px; color: #FF6347; margin-top: 20px; }
    .report { font-size: 24px; color: #FF6347; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------#
#         Sidebar Navigation    #
# -----------------------------#
st.sidebar.markdown("""
    <h2 style="font-size: 30px; font-family: 'Product Sans', sans-serif;text-align: center; color: #23272F;">Computer Vision Project</h2>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", ["Home", "Developers"])

# -----------------------------#
#         Page: Home            #
# -----------------------------#
if page == "Home":
    st.markdown('<h1 class="main-title">Cross-modal Memory Networks for Radiology Report Generation</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="custom-text">Upload X-rays to receive an AI-driven classification and report.</p>',
                unsafe_allow_html=True)

    front_view_file = st.file_uploader(
        "", type=["jpg", "jpeg", "png"], key="front")

    # Preprocessing function (same as used in DataLoader)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    def process_and_display_image(image_file, label):
        # Check if the file is valid
        if image_file is not None:
            try:
                # Open the image as a PIL object
                pil_image = Image.open(image_file).convert('RGB')
                print(f"Image loaded: {pil_image}")  # Log image loading

                # Apply transformations to get the PyTorch tensor for model input
                transformed_image = transform(pil_image)
                # Log transformed image
                print(f"Transformed image shape: {transformed_image.shape}")

                # Save the original PIL image to a buffer for Base64 encoding
                buffer = BytesIO()
                pil_image.save(buffer, format="PNG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode()

                # Display the image
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <h4>{label}</h4>
                        <img src="data:image/png;base64,{image_base64}" alt="Uploaded Image" style="border-radius: 10px; max-width: 100%; height: auto;" />
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Return the tensorized image for model inference
                return transformed_image
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return None
        else:
            st.error("No file uploaded!")
            return None

    # Check if front_image is None before proceeding with model inference
    front_image = process_and_display_image(front_view_file, "Uploaded Image")

    # Only proceed if front_image is not None
    if front_view_file is not None:
        try:
            # Open the file to check its format
            img = Image.open(front_view_file)
            format = img.format.lower()  # Get the format (e.g., "jpeg", "png")

            # Determine the output file name based on the detected format
            uploaded_image_path = f"uploaded_image.{format}"

            # Save the file in its original format
            img.save(uploaded_image_path)

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")

        n_gpu_use = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')

        # Example usage of the Args class
        args = Args()

        # Initialize the Tokenizer (you can skip 'args' if you're not training)
        tokenizer = Tokenizer(args)

        # Initialize the model
        model = BaseCMNModel(args=args, tokenizer=tokenizer).to(device)

        # Load the model weights from the saved `.pth` file (checkpoint)
        model_path = 'model_best.pth'  # Path to your saved .pth model file

        # Print log message for loading checkpoint
        log_message = f"Loading checkpoint: {model_path} ..."
        print(log_message)  # Print to console
        # self.logger.info(log_message)  # Optional, for logging to file if needed

        # Load checkpoint
        checkpoint = torch.load(
            model_path, map_location=torch.device('cpu'))  # Load checkpoint
        # Load the model state_dict
        model.load_state_dict(checkpoint['state_dict'])

        model.eval()  # Set the model to evaluation mode

        # Define the transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor(),          # Convert image to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])  # Normalize
        ])

        # Preprocess the image
        def preprocess_image(image_path):
            image = Image.open(image_path).convert(
                'RGB')  # Load image in RGB format
            return transform(image)  # Apply the transformations

        # Load and preprocess the image
        preprocessed_image = preprocess_image(uploaded_image_path)

        # Add a batch dimension
        images = preprocessed_image.unsqueeze(
            0).to(device)  # Shape: [1, C, H, W]

        # Make the prediction
        with torch.no_grad():
            # Pass the image to the model for inference
            output, _ = model(images, mode='sample')
            decoded_output = tokenizer.decode_batch(
                output.cpu().numpy())  # Decode the output to text

        # Load the model
        device = torch.device("cpu")  # Change to "cuda" if using GPU
        model_save_path = "combined_model.pth"
        loaded_model = MLPClassifier(input_dim=4608, output_dim=16)
        loaded_model.load_state_dict(torch.load(
            model_save_path, map_location=device))
        loaded_model.eval()  # Set the model to evaluation mode

        # Load the label encoder
        label_encoder_save_path = "label_encoder.pkl"
        loaded_label_encoder = joblib.load(label_encoder_save_path)

        # Initialize autoencoder
        ae = xrv.autoencoders.ResNetAE(weights="101-elastic").to(device)

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
            # Get the encoded features from the autoencoder
            z = ae.encode(img_tensor)
            # Flatten the feature map to 1D
            flattened_features = z.view(z.size(0), -1)
            # Remove batch dimension and convert to numpy
            return flattened_features.squeeze().detach().cpu().numpy()

        # Prediction function
        def predict(model, image_path, label_encoder, device):
            # Preprocess the input image and extract features
            input_data = preprocess_input(image_path, device)

            # Convert the features to a tensor
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(
                0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor.to(device))  # Forward pass
                # Apply softmax to get probabilities
                probabilities = F.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(
                    probabilities, dim=1).item()  # Get the predicted class index

            # Decode the label
            predicted_label = label_encoder.inverse_transform(
                [predicted_class_idx])[0]
            # Return the predicted label and the probabilities
            return predicted_label, probabilities.cpu().numpy()

        # Perform prediction
        try:
            predicted_label, probabilities = predict(
                loaded_model, uploaded_image_path, loaded_label_encoder, device
            )
            st.markdown(
                f'<h4 class="prediction">Predicted Label: {predicted_label}</h4>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

        st.markdown(
            f'<h4 class="report">Report: <strong>{decoded_output[0]}</strong></h4>', unsafe_allow_html=True)

# -----------------------------#
#         Page: About           #
# -----------------------------#
elif page == "Developers":
    st.markdown('<h1 class="main-title">About Us</h1>', unsafe_allow_html=True)
    team_path = os.path.join('image', "team.JPG")
    if os.path.exists(team_path):
        st.image(team_path)
