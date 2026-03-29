import torch

def handle_missing(image, clinical, text_embedding):

    if image is None:
        image = torch.zeros((1, 256))

    if clinical is None:
        clinical = torch.zeros((1, 256))

    if text_embedding is None:
        text_embedding = torch.zeros((1, 256))

    return image, clinical, text_embedding