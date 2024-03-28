import os
import cv2
import numpy as np
import easyocr
from flask import Flask, render_template, request
from PIL import Image
from autocorrect import Speller

app = Flask(__name__)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Initialize spell checker
spell = Speller()

@app.route('/')
def index():
    return render_template('index.html')

def process_frame(frame):
    # Convert frame to PIL Image
    pil_image = Image.fromarray(frame)

    # Convert to grayscale
    gray_image = pil_image.convert('L')

    # Convert back to OpenCV format
    opencv_image = np.array(gray_image)

    return opencv_image

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file:
        # Read the image file
        nparr = np.fromstring(file.read(), np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the image
        processed_image = process_frame(image_np)

        # Perform OCR
        text_results = reader.readtext(processed_image)

        # Perform spelling correction
        corrected_text = ''
        for result in text_results:
            corrected_text += spell(result[1]) + '\n'

        return render_template('index.html', detected_text=corrected_text)

if __name__ == '__main__':
    app.run(debug=True)
