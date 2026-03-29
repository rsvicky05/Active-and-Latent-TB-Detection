import numpy as np

def explain_clinical(model, clinical_tensor):

    # Convert tensor to numpy
    clinical_values = clinical_tensor.detach().cpu().numpy()[0]

    cbc = clinical_values[0]
    esr = clinical_values[1]
    crp = clinical_values[2]

    # Simple rule-based importance scoring
    importance = {
        "CBC": round(abs(cbc) * 0.1, 3),
        "ESR": round(abs(esr) * 0.5, 3),
        "CRP": round(abs(crp) * 1.0, 3)
    }

    return importance