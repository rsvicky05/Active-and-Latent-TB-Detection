import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(attn_weights):

    weights = attn_weights.mean(dim=1).detach().cpu().numpy()

    plt.imshow(weights[0], cmap="viridis")
    plt.colorbar()
    plt.title("Modality Attention Map")
    plt.show()