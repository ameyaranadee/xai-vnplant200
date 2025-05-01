import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, render_template, jsonify, request
import tensorflow as tf

from webapp.utils import vit_transform
from webapp.models import InceptionModel, VisionTransformerModel
from webapp.explainers import LimeExplainer, GradCamExplainer
from webapp.segmentation import Segmentation

app = Flask(__name__, template_folder='templates')

# initialize models and explainers
inceptionv3_model = InceptionModel('../models/inceptionv3')
vit_model = VisionTransformerModel()
vit_explainer = GradCamExplainer(vit_model.model)
lime_explainer = LimeExplainer(inceptionv3_model)
segmentation = Segmentation()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segmentation', methods=['GET'])
def get_segmentation():
    ratio = float(request.args.get('ratio', 1.0))
    max_dist = float(request.args.get('max_dist', 10.0))
    image_path = plt.imread('static/images/5.JPG')
    segments = segmentation.apply_quickshift(image_path, ratio, max_dist)    
    segmentation_data = [segmentation.image_to_base64(segment) for segment in segments]

    return jsonify(segmentation_data)

@app.route('/explain', methods=['POST'])
def explain():
    image_data = request.files['image'].read()
    model_type = request.form.get('model', 'vit')
    explanation_images = []

    if model_type == 'vit':
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = vit_transform(image).unsqueeze(0)
        top_classes, top_probs = vit_model.predict(input_tensor)
        
        for cls in top_classes:
            explanation = vit_explainer.explain(input_tensor, cls)
            fig, ax = plt.subplots()
            ax.imshow(explanation)
            ax.axis('off')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            explanation_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            explanation_images.append(explanation_image_base64)

    elif model_type == 'inception':
        raw_img = inceptionv3_model.load_image(image_data)
        img = inceptionv3_model.transform_img(raw_img)
        top_classes, top_probs = inceptionv3_model.predict(img)

        num_features = int(request.form.get('num_features', 5))
        positive_only = request.form.get('positive_only', 'false') == 'true'
        hide_rest = request.form.get('hide_rest', 'false') == 'true'

        for _ in top_classes:
            # explanation = lime_explainer.explain(img)
            _, explanation = lime_explainer.explain(img, num_features, positive_only, hide_rest)
            buffer = io.BytesIO()
            plt.imsave(buffer, explanation, format='png')
            explanation_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            explanation_images.append(explanation_image_base64)
            # predicted_class, explanation_image = explainer.explain(img)
            # explanation_image_base64 = segmentation.image_to_base64(explanation_image)
            # processed_image_path = 'static/images/explanation.png'
            # plt.imsave(processed_image_path, explanation_image, format='png')
    
    else:
        return jsonify({'error': 'Unsupported model selected'}), 400
    
    print('Top classes:', top_classes)
    print('Top probabilities:', top_probs)

    return jsonify({
        'top_classes': [int(cls) for cls in top_classes],
        'top_probs': [float(p) for p in top_probs],
        'explanations': explanation_images
    })

@app.route('/adjust_parameters', methods=['POST'])
def adjust_parameters():
    num_features = int(request.form['num_features'])
    positive_only = request.form.get('positive_only') == 'true'
    hide_rest = request.form.get('hide_rest') == 'true'

    # Re-explain with adjusted parameters
    img = inceptionv3_model.load_image(open('static/images/5.JPG', "rb").read())
    img = inceptionv3_model.transform_img(img)

    # Generate a new explanation with adjusted parameters
    predicted_class, new_explanation_image = lime_explainer.explain(img, num_features, positive_only, hide_rest)

    # Save the adjusted image
    new_image_path = 'static/images/new_explanation.png'
    plt.imsave(new_image_path, new_explanation_image, format='png')

    new_explanation_image_base64 = segmentation.image_to_base64(new_explanation_image)
    
    # Return the predicted class and explanation image as a JSON response
    return jsonify({'new_explanation_image': new_explanation_image_base64})

if __name__ == '__main__':
    app.run(debug=True)