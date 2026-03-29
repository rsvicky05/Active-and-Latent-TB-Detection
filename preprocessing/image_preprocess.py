import torchvision.transforms as transforms
from PIL import Image

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path):
    transform = get_image_transform()
    image = Image.open(image_path).convert("RGB")
    return transform(image)