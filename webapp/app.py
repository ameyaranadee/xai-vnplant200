from flask import Flask, render_template, jsonify, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift
import io
import base64
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load the model from the 'inceptionv3' directory
loaded_inceptionv3_model = load_model('../inceptionv3')

def LoadImage(image_data):
    # Load image from the image data
    image_file = io.BytesIO(image_data)
    image = skimage.io.imread(image_file)
    return image

def transform_img(img):
    img = skimage.transform.resize(img, (150, 150))
    img = (img - 0.5) * 2
    img = np.expand_dims(img, axis=0)
    return img

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segmentation', methods=['GET'])
def get_segmentation():
    ratio = float(request.args.get('ratio', 1.0))
    max_dist = float(request.args.get('max_dist', 10.0))
    
    # Load the input image
    train_image_orig = plt.imread('static/5.JPG')
    
    # Generate the segmentation results
    segments = quickshift(train_image_orig, ratio=ratio)
    segments1 = quickshift(train_image_orig, max_dist=max_dist)
    
    # Convert the segmentation results to base64-encoded images
    segmentation_data = []
    for segment in [train_image_orig, segments, segments1]:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(segment)
        ax.axis('off')
        
        # Save the image to a buffer and convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        segmentation_data.append(img_base64)
    
    return jsonify(segmentation_data)

@app.route('/explain', methods=['POST'])
def explain():
    # Get the image data from the request
    image_data = request.files['image'].read()
    
    # Load the image
    img = LoadImage(image_data)
    
    # Preprocess the image
    img = transform_img(img)
    
    # Create the LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Generate the LIME explanation
    explanation = explainer.explain_instance(img[0].astype('double'), loaded_inceptionv3_model.predict, top_labels=3, hide_color=0, num_samples=1000)
    
    # Get the predicted class
    predicted_class = explanation.top_labels[0]
    
    # Generate the explanation image
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    explanation_image = mark_boundaries(temp / 2 + 0.5, mask)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(segment)
    ax.axis('off')

    # Convert the explanation image to base64
    buffer = io.BytesIO()
    plt.imsave(buffer, explanation_image, format='png')
    explanation_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Return the predicted class and explanation image as a JSON response
    return jsonify({'predicted_class': str(predicted_class), 'explanation_image': explanation_image_base64})

if __name__ == '__main__':
    app.run(debug=True)