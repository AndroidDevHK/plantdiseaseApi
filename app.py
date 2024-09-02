from flask import Flask, request, jsonify, send_from_directory
import os
from PIL import Image
from ultralytics import YOLO
import time

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('best_float32.tflite', task='segment')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save the file
        input_image_path = 'input_' + file.filename
        file.save(input_image_path)
        
        # Process the image
        start = time.time()
        results = model(input_image_path)
        run_time = time.time() - start

        # Process results
        highest_confidence = {}
        output_image_name = 'output_' + file.filename

        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            
            for det in r.boxes:
                if hasattr(det, 'cls') and hasattr(det, 'conf'):
                    class_id = int(det.cls)
                    confidence = det.conf.item()
                    
                    class_names = ["Early_blight", "Late_blight", "Leaf"]
                    class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"
                    
                    if class_name not in highest_confidence or confidence > highest_confidence[class_name]:
                        highest_confidence[class_name] = confidence
            
            im.save(output_image_name)
        
        # Return results
        response = {
            "run_time": f"{run_time:.2f} sec",
            "results": highest_confidence,
            "image": output_image_name
        }
        
        return jsonify(response)

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True)
