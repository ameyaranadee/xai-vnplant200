import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from webapp.utils import reshape_transform

class LimeExplainer:
    def __init__(self, model):
        self.model = model
    
    def explain(self, img, num_features=5, positive_only=False, hide_rest=False):
        explainer = lime_image.LimeImageExplainer()
        
        # Generate the LIME explanation
        explanation = explainer.explain_instance(img[0].astype('double'), self.model.batch_predict, top_labels=3, hide_color=0, num_samples=1000)
        predicted_class = explanation.top_labels[0]
        # Generate the explanation image
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=positive_only, num_features=num_features, hide_rest=hide_rest)
        explanation_image = mark_boundaries(temp / 2 + 0.5, mask)

        return predicted_class, explanation_image
    
class GradCamExplainer:
    def __init__(self, model):
        self.model = model
        # self.target_layers = [model.blocks[-1].norm1]
        self.target_layers = [model.norm]
        # self.explainer = GradCAMPlusPlus(model=model, target_layers=self.target_layers)
        self.cam_constructor = GradCAM
        self.show_cam_on_image = show_cam_on_image
        self.ClassifierOutputTarget = ClassifierOutputTarget

    def explain(self, input_tensor, target_class):
        # cam_map = self.explainer(input_tensor=input_tensor, targets=[torch.nn.functional.one_hot(torch.tensor([target_class]), num_classes=200).float().to(input_tensor.device)])
        # grayscale_cam = cam_map[0, :]
        # return visualize_cam(grayscale_cam, input_tensor.squeeze().permute(1, 2, 0).cpu().numpy())
        rgb_img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())  # normalize to [0, 1]

        # with self.cam_constructor(model=self.model, target_layers=self.target_layers, use_cuda=torch.cuda.is_available()) as cam:
        #     grayscale_cam = cam(input_tensor=input_tensor, targets=[self.ClassifierOutputTarget(target_class)])[0]
        #     visualization = self.show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        #     return visualization
        cam = self.cam_constructor(model=self.model, target_layers=self.target_layers, reshape_transform=reshape_transform)
        cam.batch_size = 1  # prevent memory leak

        grayscale_cam = cam(input_tensor=input_tensor, targets=[self.ClassifierOutputTarget(target_class)])[0]
        visualization = self.show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return visualization