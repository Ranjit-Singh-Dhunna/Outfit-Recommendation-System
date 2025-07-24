import os
import time
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from roboflow import Roboflow
from uuid import uuid4
import cv2
import numpy as np
from sklearn.cluster import KMeans
import json
from typing import List, Dict
from dotenv import load_dotenv
import requests
from PIL import Image
import io
import google.generativeai as genai
import logging
import cv2
import numpy as np
import requests
from typing import Tuple
import mediapipe as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load environment variables
load_dotenv('keys.env')

# Initialize Roboflow

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
ROBOFLOW_WORKSPACE = os.getenv('ROBOFLOW_WORKSPACE')
ROBOFLOW_PROJECT = os.getenv('ROBOFLOW_PROJECT')
ROBOFLOW_VERSION = os.getenv('ROBOFLOW_VERSION')
if not ROBOFLOW_API_KEY:
    print("Error: ROBOFLOW_API_KEY not found in .env file")
    exit(1)

try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)   
    clothing_model = (
        rf.workspace(os.getenv('ROBOFLOW_WORKSPACE'))
        .project(os.getenv('ROBOFLOW_PROJECT'))
        .version(int(os.getenv('ROBOFLOW_VERSION')))
        .model
    )
except Exception as e:
    print(f"Failed to initialize Roboflow: {str(e)}")
    exit(1)

# Initialize Gemini
GEMINI_WORKING = False
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model_name = 'models/gemini-1.5-flash'
        print(f"Using model: {model_name}")
        gemini = genai.GenerativeModel(model_name)
        GEMINI_WORKING = True
    else:
        print("Warning: GEMINI_API_KEY not found in .env file")
except Exception as e:
    print(f"Gemini initialization failed: {str(e)}")

def create_composite_image(image_paths: List[str], output_path: str, max_height: int = 500) -> str:
    """Create a horizontal composite image with consistent dimensions"""
    try:
        images = []
        for path in image_paths:
            img = Image.open(path)
            images.append(img)
        
        if not images:
            return ""
            
        # Calculate new widths maintaining aspect ratio
        heights = [img.height for img in images]
        min_height = min(heights)
        if max_height < min_height:
            min_height = max_height
            
        resized_images = []
        total_width = 0
        for img in images:
            aspect_ratio = img.width / img.height
            new_width = int(min_height * aspect_ratio)
            resized_img = img.resize((new_width, min_height), Image.Resampling.LANCZOS)
            resized_images.append(resized_img)
            total_width += new_width
            
        # Create new composite image
        composite = Image.new('RGB', (total_width, min_height))
        x_offset = 0
        for img in resized_images:
            composite.paste(img, (x_offset, 0))
            x_offset += img.width
            
        composite.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error creating composite image: {str(e)}")
        return ""

def remove_background(img: np.ndarray) -> np.ndarray:
    """Improved background removal using adaptive thresholding"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.bitwise_and(img, img, mask=mask)
    except:
        return img

def extract_dominant_color(image_path: str) -> str:
    """Robust color extraction with background removal"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "#000000"
            
        img = remove_background(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation_mask = cv2.inRange(hsv[:,:,1], 50, 255)
        
        pixels = img.reshape(-1, 3)
        sat_mask = saturation_mask.reshape(-1)
        colored_pixels = pixels[sat_mask > 0]
        
        if len(colored_pixels) == 0:
            return "#000000"
            
        kmeans = KMeans(n_clusters=3, n_init=10)
        kmeans.fit(colored_pixels)
        dominant_color = kmeans.cluster_centers_[
            np.argmax(np.bincount(kmeans.labels_))
        ].astype(int)
        
        return f"#{dominant_color[2]:02X}{dominant_color[1]:02X}{dominant_color[0]:02X}"
    except Exception as e:
        print(f"Color extraction error: {str(e)}")
        return "#000000"

def process_images(image_paths: List[str]) -> List[Dict]:
    """Process multiple images and return combined metadata"""
    all_metadata = []
    
    for image_path in image_paths:
        try:
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue
                
            image_id = str(uuid4())
            metadata = {
                "image_id": image_id,
                "detections": [],
                "source_image": os.path.basename(image_path)
            }

            # ✅ Add confidence/overlap explicitly and handle API errors
            try:
                print(f"Sending to Roboflow: {image_path}")
                prediction = clothing_model.predict(image_path, confidence=40, overlap=30)
                clothing_results = prediction.json()
            except Exception as e:
                print(f"❌ Roboflow failed on {image_path}: {str(e)}")
                continue
            
            for detection in clothing_results.get("predictions", []):
                try:
                    item = detection["class"]
                    bbox = [
                        detection["x"],
                        detection["y"],
                        detection["width"],
                        detection["height"]
                    ]
                    
                    x, y, w, h = map(int, bbox)
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"cv2 failed to read {image_path}")
                        continue
                        
                    cropped_img = img[y:y+h, x:x+w]
                    temp_path = f"temp_{image_id}_{item}.jpg"
                    cv2.imwrite(temp_path, cropped_img)
                    
                    dominant_color = extract_dominant_color(temp_path)
                    
                    metadata["detections"].append({
                        "item": item,
                        "color_hex": dominant_color,
                        "bbox": bbox
                    })
                    os.remove(temp_path)
                    
                except Exception as e:
                    print(f"⚠️ Skipping detection in {image_path}: {str(e)}")
                    continue
                    
            all_metadata.append(metadata)
        except Exception as e:
            print(f"❌ Failed to process {image_path}: {str(e)}")
    
    return all_metadata


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'drip_html.html')

@app.route('/api/upload', methods=['POST'])
def upload_images():
    print("Upload endpoint hit")
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({"error": "No files selected"}), 400
    
    image_paths = []
    for file in files:
        if file.filename == '':
            continue
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image_paths.append(filepath)
    
    if not image_paths:
        return jsonify({"error": "No valid images uploaded"}), 400
    
    try:
        metadata = process_images(image_paths)
        return jsonify({
            "success": True,
            "metadata": metadata,
            "message": f"Processed {len(metadata)} images"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Failed to process images"
        }), 500

@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    print("Suggestions endpoint hit")
    try:
        data = request.get_json()
        print("Received data:", data)
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
            
        required = ['metadata', 'userProfile']
        if not all(field in data for field in required):
            return jsonify({"success": False, "error": f"Missing required fields: {required}"}), 400
        
        metadata = data['metadata']
        user_profile = data['userProfile']
        
        
        if GEMINI_WORKING:
            try:
                prompt = f"""
                Wardrobe Items:
                {json.dumps(metadata, indent=2)}
                
                User Profile:
                - Gender: {user_profile['gender']}
                - Skin tone: {user_profile['skinTone']}
                - Body type: {user_profile['bodyType']}
                - Style: {user_profile['style']}
                - Occasion: {user_profile['occasion']}
                - Facial Hair: {user_profile['facial']}
                
                Suggest {user_profile['num']} personalized outfit combinations. For each outfit:
                1. List the specific items used (reference their EXACT image filenames)
                2. Provide detailed reasoning in 2-3 sentences
                3. Return ONLY pure JSON formatted EXACTLY like this:
                {{
                    "outfits": [
                        {{
                            "outfit_name": "name",
                            "items": ["exact_filename1.jpg", "exact_filename2.jpg"],
                            "reasoning": "text"
                        }}
                    ]
                }}
                {user_profile['message']}
                If there is any note by you , include it in each "reasoning" you generated.
Here is a list of items with both filename and description. Use  the filenames in the outfit suggestions and  include the description in reasoning.
                """
                
                response = gemini.generate_content(prompt)
                print("Gemini raw response:", response.text)
                
                try:
                    json_str = response.text.replace('```json', '').replace('```', '').strip()
                    response_data = json.loads(json_str)
                    
                    if not isinstance(response_data.get('outfits'), list):
                        raise ValueError("Response does not contain outfits list")
                        
                    # Create composite images for each outfit
                    for outfit in response_data['outfits']:
                        image_paths = [
                            os.path.join(app.config['UPLOAD_FOLDER'], item) 
                            for item in outfit['items']
                            if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], item))
                        ]
                        if image_paths:
                            composite_filename = f"composite_{uuid4().hex}.jpg"
                            composite_path = os.path.join('static', 'composites', composite_filename)
                            os.makedirs(os.path.dirname(composite_path), exist_ok=True)
                            create_composite_image(image_paths, composite_path)
                            outfit['composite_image'] = f"/static/composites/{composite_filename}"
                    
                    return jsonify({
                        "success": True,
                        "outfits": response_data['outfits'],
                        "aiUsed": True
                    })
                    
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Failed to parse Gemini response: {str(e)}")
                    print("Response content:", response.text)
                    outfits = generate_fallback_suggestions_with_images(metadata)
                    
            except Exception as e:
                print(f"Gemini error: {str(e)}")
                outfits = generate_fallback_suggestions_with_images(metadata)
        else:
            outfits = generate_fallback_suggestions_with_images(metadata)
        
        return jsonify({
            "success": True,
            "outfits": outfits,
            "aiUsed": GEMINI_WORKING
        })
        
    except Exception as e:
        print(f"Error in suggestions endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to generate suggestions"
        }), 500

def generate_fallback_suggestions_with_images(metadata):
    """Fallback that includes image references and creates composite images"""
    suggestions = []
    
    items_with_images = []
    for img_data in metadata:
        for item in img_data["detections"]:
            items_with_images.append({
                "item": item["item"],
                "color": item["color_hex"],
                "image": img_data["source_image"]
            })
    
    if len(items_with_images) >= 2:
        image_paths = [
            os.path.join(app.config['UPLOAD_FOLDER'], items_with_images[0]["image"]),
            os.path.join(app.config['UPLOAD_FOLDER'], items_with_images[1]["image"])
        ]
        composite_filename = f"composite_{uuid4().hex}.jpg"
        composite_path = os.path.join('static', 'composites', composite_filename)
        os.makedirs(os.path.dirname(composite_path), exist_ok=True)
        create_composite_image(image_paths, composite_path)
        
        suggestions.append({
            "outfit_name": "Casual Work Combo",
            "items": [items_with_images[0]["image"], items_with_images[1]["image"]],
            "reasoning": "This combination pairs neutral colors for a professional yet comfortable look.",
            "composite_image": f"/static/composites/{composite_filename}"
        })
        
    if len(items_with_images) >= 3:
        image_paths = [
            os.path.join(app.config['UPLOAD_FOLDER'], items_with_images[1]["image"]),
            os.path.join(app.config['UPLOAD_FOLDER'], items_with_images[2]["image"])
        ]
        composite_filename = f"composite_{uuid4().hex}.jpg"
        composite_path = os.path.join('static', 'composites', composite_filename)
        create_composite_image(image_paths, composite_path)
        
        suggestions.append({
            "outfit_name": "Color Pop Outfit",
            "items": [items_with_images[1]["image"], items_with_images[2]["image"]],
            "reasoning": "Bold color pairing that makes a statement while keeping it work-appropriate.",
            "composite_image": f"/static/composites/{composite_filename}"
        })
    
    if len(items_with_images) >= 4:
        image_paths = [
            os.path.join(app.config['UPLOAD_FOLDER'], items_with_images[0]["image"]),
            os.path.join(app.config['UPLOAD_FOLDER'], items_with_images[3]["image"])
        ]
        composite_filename = f"composite_{uuid4().hex}.jpg"
        composite_path = os.path.join('static', 'composites', composite_filename)
        create_composite_image(image_paths, composite_path)
        
        suggestions.append({
            "outfit_name": "Versatile Mix",
            "items": [items_with_images[0]["image"], items_with_images[3]["image"]],
            "reasoning": "A balanced combination that works for multiple occasions.",
            "composite_image": f"/static/composites/{composite_filename}"
        })
    
    return suggestions

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/uploads/<path:filename>')
def uploaded_files(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/test')
def test_route():
    return jsonify({"message": "API is working!", "routes": [
        "/api/upload (POST)",
        "/api/suggestions (POST)",
        "/api/virtual-tryon (POST)",
        "/static/<path> (GET)",
        "/uploads/<path> (GET)"
    ]})

if __name__ == '__main__':
    # Create composites directory if it doesn't exist
    os.makedirs(os.path.join('static', 'composites'), exist_ok=True)
    
    print("Starting server with routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.rule} ({', '.join(rule.methods)})")
    app.run(host='0.0.0.0', port=5002, debug=True)