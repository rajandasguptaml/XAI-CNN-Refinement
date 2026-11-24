"""Explainability-driven CNN refinement for brain tumor classification in MRI.

Code for

    Rajan Das Gupta, Md Imrul Hasan Showmick, Lei Wei, Mushfiqur Rahman Abir, Shanjida Akter, Md. Yeasin Rahat,
    Md. Jakir Hossen. "From Explanations to Architecture: Explainability-Driven CNN Refinement for Brain Tumor
    Classification in MRI." BIC 2026. https://doi.org/10.1145/3809986.3810119

Modules
-------
preprocess  brain-region cropping, intensity normalisation, resizing, image cache
data        dataset discovery, the paper's random split and the leak-free group-aware split, external test sets
model       configurable Conv(3x3)+ReLU+MaxPool CNN
gradcam     Grad-CAM heatmaps and layer-wise relevance m_l (Eq. 4)
refine      percentile threshold tau (Eq. 5) and block removal
train       training loop (Adam, lr 1e-3, batch 40, 40 epochs)
metrics     accuracy, macro precision / recall / F1, confusion matrix
explain     Grad-CAM, SHAP and LIME panels
"""

__version__ = "0.1.0"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
