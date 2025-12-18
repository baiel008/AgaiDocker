import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from PIL import Image

# ---------- CLASSES ----------
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ---------- MODEL ----------
class CifarClassifaction(nn.Module):  # имена как в старой модели
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# ---------- LOAD MODEL ----------
device = torch.device("cpu")
model = CifarClassifaction()

model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ---------- APP ----------
app = FastAPI(title="CIFAR-10 Classifier")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            class_id = output.argmax(dim=1).item()

        return {
            "class_id": class_id,
            "class_name": CLASSES[class_id]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
