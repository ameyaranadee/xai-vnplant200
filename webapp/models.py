import io
import torch
import timm
import skimage.io
import numpy as np
import tensorflow as tf
import skimage.transform
from keras.layers import TFSMLayer

class InceptionModel:
    def __init__(self, model_path):
        # self.model = load_model(model_path)
        self.model = TFSMLayer(model_path, call_endpoint='serving_default')

    def transform_img(self, img):
        img = skimage.transform.resize(img, (150, 150))
        img = (img - 0.5) * 2
        img = np.expand_dims(img, axis=0)
        return img
    
    def load_image(self, image_data):
        # Load image from the image data
        image_file = io.BytesIO(image_data)
        image = skimage.io.imread(image_file)
        return image

    def predict(self, img):
        # return self.model.predict(img)
        # probs = self.model.predict(img)[0]
        print('Image shape:', img.shape)
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
        images = tf.convert_to_tensor(img, dtype=tf.float32)
        output_dict = self.model(images)
        probs = output_dict['flatten'].numpy()
        # probs = output_tensor.numpy()[0]
        row = probs[0]
        top_classes = np.argsort(row)[-3:][::-1]
        print('Top classes:', top_classes)
        top_probs = row[top_classes]
        return top_classes, top_probs
    
    def batch_predict(self, imgs):
        # For LIME: returns probabilities from all samples in the batch.
        if len(imgs.shape) == 3:
            imgs = np.expand_dims(imgs, axis=0)
        images = tf.convert_to_tensor(imgs, dtype=tf.float32)
        output_dict = self.model(images)
        probs = output_dict['flatten'].numpy()
        return probs
    
# Vision Transformer
class VisionTransformerModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=200)
        self.model.to(self.device).eval()

    def predict(self, img_tensor):
        with torch.no_grad():
            outputs = self.model(img_tensor.to(self.device))
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_classes = torch.topk(probs, 3)
            return top_classes.cpu().numpy(), top_probs.cpu().numpy()
