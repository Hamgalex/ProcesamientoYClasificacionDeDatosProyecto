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

        # Transformaciones (idénticas a las del entrenamiento)
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

    # ==============================================================
    # 🧼 LIMPIEZA DE LA RADIOGRAFÍA (solo huesos)
    # ==============================================================
    def _preprocess_bones(self, image_path):
        """Limpia la radiografía para dejar solo los huesos visibles."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 1️⃣ Aumentar contraste
        img_eq = cv2.equalizeHist(img)

        # 2️⃣ Reducir ruido pero mantener bordes
        img_blur = cv2.bilateralFilter(img_eq, d=9, sigmaColor=75, sigmaSpace=75)

        # 3️⃣ Umbral adaptativo para destacar huesos (zonas blancas)
        _, thresh = cv2.threshold(img_blur, 130, 255, cv2.THRESH_BINARY)

        # 4️⃣ Limpieza morfológica
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 5️⃣ Aplicar máscara sobre imagen original en color
        img_color = cv2.imread(image_path)
        result = cv2.bitwise_and(img_color, img_color, mask=mask)

        # Guardar versión limpia
        clean_path = os.path.join(os.path.dirname(image_path),
                                  "clean_" + os.path.basename(image_path))
        cv2.imwrite(clean_path, result)

        return clean_path

    # ==============================================================
    # 🧠 PREDICCIÓN PRINCIPAL
    # ==============================================================
    def predict(self, image_path):
        # Primero limpiar la imagen
        clean_path = self._preprocess_bones(image_path)

        # Cargar imagen limpia y aplicar transformaciones
        image = Image.open(clean_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, preds = torch.max(outputs, 1)

        predicted_label = self.classes[preds.item()]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds.item()].item()

        result = {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "clean_image_path": clean_path
        }

        # Si está fracturada, generar Grad-CAM
        if predicted_label == "fractured":
            result["heatmap_path"] = self._generate_heatmap(clean_path, input_tensor)

        return result

    # ==============================================================
    # 🔥 GRAD-CAM
    # ==============================================================
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
