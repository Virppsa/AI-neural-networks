from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
import io
from PIL import Image
import base64
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


class ComplexConvNet(torch.nn.Module):
    def __init__(self, in_shape, out_classes):
        super().__init__()
        self.conv1_1 = torch.nn.Conv2d(in_shape[0], 16, 3, padding='same')
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding='same')
        self.pool1 = torch.nn.MaxPool2d((2, 2), (2, 2))
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding='same')
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding='same')
        self.pool2 = torch.nn.MaxPool2d((2, 2), (2, 2))
        self.fc3 = torch.nn.Linear(32 * in_shape[1] * in_shape[2] // 4**2, 128)
        self.fc4 = torch.nn.Linear(128, out_classes)

    def forward(self, x):
        y = torch.nn.Sequential(
            self.conv1_1,
            torch.nn.ReLU(),
            self.conv1_2,
            torch.nn.ReLU(),
            self.pool1,
            self.conv2_1,
            torch.nn.ReLU(),
            self.conv2_2,
            torch.nn.ReLU(),
            self.pool2,
            torch.nn.Flatten(),
            self.fc3,
            torch.nn.ReLU(),
            self.fc4
        )(x)
        return y


class_names = ['dog', 'cat', 'horse']
app = Flask(__name__)
api = Api(app)
CORS(app, resources={r"*": {"origins": "*"}})

# Uzloadiname modeli is modelio failo
model = ComplexConvNet(torch.Size([3, 128, 128]), 3)
model_weights = torch.load(
    './model/image_predictor.pth', map_location=torch.device('cpu'))
model.load_state_dict(model_weights)
model = model.to('cpu')
model.eval()


def preprocess_image(image_file):
    image = Image.open(image_file)
    transformed_image = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])(image)
    return transformed_image.unsqueeze(0)

# Endpointo klase


class Prediction(Resource):
    def post(self):
        try:
            # Gauname paveiksleli is requesto
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "Nedavete paveikslelio"}), 400

            # Procesuojame paveiksleli
            processed_image = preprocess_image(file)
            processed_image = processed_image.to('cpu')

            # Darome predictiona pagal uzloadinta modeli
            with torch.no_grad():
                output = model(processed_image)
                output_softmax = F.softmax(output, dim=1)
                conf, predicted_idx = torch.max(output_softmax.data, 1)

            response = {
                "class_name": class_names[predicted_idx.item()],
                "confidence": f"{conf.item() * 100:.2f}%"
            }

        except Exception as e:
            print("Error processing request:", e)
            return jsonify({"error": str(e)}), 500

        return jsonify(response)


api.add_resource(Prediction, '/predict')
app.run(port=5000)

# Startuoti: python3 api.py
