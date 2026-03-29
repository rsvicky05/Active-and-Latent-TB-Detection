import torch
import cv2
import numpy as np
import os

def generate_gradcam(model, image_tensor, clinical, image_path):

    model.eval()

    gradients = []
    activations = []

    # Hook the LAST CONV LAYER of ResNet
    target_layer = model.layer4[-1].conv2

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)

    class_idx = output.argmax(dim=1)

    model.zero_grad()
    output[0, class_idx].backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    if len(gradients) == 0:
        raise RuntimeError("Gradients not captured.")

    grads = gradients[0]
    acts = activations[0]

    # Global Average Pooling on gradients
    weights = torch.mean(grads, dim=[2, 3], keepdim=True)

    cam = torch.sum(weights * acts, dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()

    cam = cam.detach().cpu().numpy()

    # Resize heatmap
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (224, 224))

    heatmap = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    save_path = os.path.join("static", "gradcam_result.jpg")
    cv2.imwrite(save_path, superimposed)

    return "gradcam_result.jpg"