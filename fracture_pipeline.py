import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image

class FracturePipeline:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Mismas transformaciones que en el entrenamiento
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # Cargar modelo
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.classes = ['fractured', 'not fractured']

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, preds = torch.max(outputs, 1)

        predicted_label = self.classes[preds.item()]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds.item()].item()

        result = {"prediction": predicted_label, "confidence": round(confidence, 4)}

        # Si hay fractura, genera mapa de calor
        if predicted_label == "fractured":
            result["heatmap_path"] = self._generate_heatmap(image_path, input_tensor)

        return result

    def _generate_heatmap(self, image_path, input_tensor):
        gradients = []
        activations = []

        def save_gradient(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        def save_activation(module, input, output):
            activations.append(output)

        target_layer = self.model.layer4[-1].conv3
        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(save_gradient)

        output = self.model(input_tensor)
        class_idx = torch.argmax(output)
        score = output[0, class_idx]

        self.model.zero_grad()
        score.backward()

        grads = gradients[0].cpu().data.numpy()[0]
        acts = activations[0].cpu().data.numpy()[0]
        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        superimposed = np.uint8(0.4 * heatmap + 0.6 * img)

        output_path = os.path.join(os.path.dirname(image_path),
                                   "heatmap_" + os.path.basename(image_path))
        cv2.imwrite(output_path, superimposed)
        return output_path
