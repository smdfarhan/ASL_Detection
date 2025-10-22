import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# ---- Device ----
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Load Model ----
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 26)
model.load_state_dict(torch.load("../models/asl_model.pth", map_location=device))
model.to(device)
model.eval()

# ---- Transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Prediction Function ----
def predict_image_top3(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)
        top3_prob, top3_idx = torch.topk(probs, 3)

    top3_letters = [chr(idx.item() + ord("A")) for idx in top3_idx[0]]
    top3_conf = [p.item()*100 for p in top3_prob[0]]

    return top3_letters, top3_conf, img

# ---- GUI to Select Image ----
root = tk.Tk()
root.withdraw()  # hide main window
file_path = filedialog.askopenfilename(title="Select an ASL Image",
                                       filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if file_path:
    top3_letters, top3_conf, img = predict_image_top3(file_path)

    # Display Image with Top-1 Prediction
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Top-1 Prediction: {top3_letters[0]} ({top3_conf[0]:.2f}%)")
    plt.show()

    # Print Top-3 Predictions
    print("Top-3 Predictions:")
    for letter, conf in zip(top3_letters, top3_conf):
        print(f"{letter}: {conf:.2f}%")
else:
    print("No file selected.")