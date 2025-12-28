import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from PIL import Image

CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

class CifarClassifaction(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            # Block 1
            conv_block(3, 64), conv_block(64, 64),
            nn.MaxPool2d(2, 2),
            # Block 2
            conv_block(64, 128), conv_block(128, 128),
            nn.MaxPool2d(2, 2),
            # Block 3
            conv_block(128, 256), conv_block(256, 256),
            conv_block(256, 256), conv_block(256, 256),
            nn.MaxPool2d(2, 2),
            # Block 4
            conv_block(256, 512), conv_block(512, 512),
            conv_block(512, 512), conv_block(512, 512),
            nn.MaxPool2d(2, 2),
            # Block 5
            conv_block(512, 512), conv_block(512, 512),
            conv_block(512, 512), conv_block(512, 512),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

device = torch.device("cpu")
model = CifarClassifaction()

model.load_state_dict(torch.load("model_cifar10.pth", map_location=device))
model.eval()

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
