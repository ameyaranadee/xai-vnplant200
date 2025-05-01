import io
import base64
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift

class Segmentation:
    @staticmethod
    def apply_quickshift(image, ratio, max_dist):
        segments_ratio = quickshift(image, ratio=ratio)
        segments_max_dist = quickshift(image, max_dist=max_dist)
        return [image, segments_ratio, segments_max_dist]

    @staticmethod
    def image_to_base64(image):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(image)
        ax.axis('off')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        return img_base64