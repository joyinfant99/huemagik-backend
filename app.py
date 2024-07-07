import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import traceback

print("Starting HueMagik backend...")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://joyinfant99.github.io", "http://localhost:3000"]}})  # Allow all origins for now, restrict in production

def get_colors(image, number_of_colors):
    try:
        # Resize image to speed up processing
        image = image.resize((150, 150))
        
        # Convert image to RGB values
        image = image.convert("RGB")
        image_array = np.array(image)
        
        # Reshape the image to be a list of pixels
        image_array = image_array.reshape((image_array.shape[0] * image_array.shape[1], 3))
        
        # Cluster and extract colors
        kmeans = KMeans(n_clusters=number_of_colors)
        kmeans.fit(image_array)
        
        # Get the colors
        colors = kmeans.cluster_centers_
        
        # Convert to integer RGB values
        colors = colors.round().astype(int)
        
        return colors.tolist()
    except Exception as e:
        print(f"Error in get_colors: {str(e)}")
        traceback.print_exc()
        return None

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        number_of_colors = int(request.form.get('colors', 5))  # Changed to 5 to match frontend
        
        # Open the image using Pillow
        image = Image.open(image_file)
        
        # Process the image and get colors
        colors = get_colors(image, number_of_colors)
        
        if colors is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        print(f"Processed colors: {colors}")  # Log the processed colors
        return jsonify({'colors': colors})
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'HueMagik Backend is working!'}), 200

if __name__ == '__main__':
    print("HueMagik backend is running!")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)