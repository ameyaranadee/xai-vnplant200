from flask import Flask, render_template, jsonify, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift
import io
import base64

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segmentation', methods=['GET'])
def get_segmentation():
    ratio = float(request.args.get('ratio', 1.0))
    
    # Load the input image
    train_image_orig = plt.imread('static/5.JPG')
    
    # Generate the segmentation results
    segments = quickshift(train_image_orig)
    segments1 = quickshift(train_image_orig, ratio=ratio)
    
    # Convert the segmentation results to base64-encoded images
    segmentation_data = []
    for segment in [train_image_orig, segments1, segments]:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(segment)
        ax.axis('off')
        
        # Save the image to a buffer and convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        segmentation_data.append(img_base64)
    
    return jsonify(segmentation_data)

if __name__ == '__main__':
    app.run(debug=True)