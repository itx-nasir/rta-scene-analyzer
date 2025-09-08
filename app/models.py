import torch
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import uuid
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# -------------------
# Load YOLOv8 (Object Detection)
# -------------------
yolo_model = YOLO("yolov8n.pt")  # Nano version (fast). Change to yolov8m.pt for higher accuracy.

def predict_objects(image_bytes, model):
    """Predict objects and draw bounding boxes, return processed image path and detections"""
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run YOLO
    results = model(img)
    detections = []
    
    # Draw bounding boxes
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cls = r
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Add detection to list
        detections.append({
            "label": model.names[int(cls)],
            "confidence": round(float(score), 3),
            "bbox": [x1, y1, x2, y2]
        })
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with confidence
        label = f"{model.names[int(cls)]}: {score:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Save processed image
    processed_filename = f"processed_{uuid.uuid4().hex[:8]}.jpg"
    processed_path = f"static/processed/{processed_filename}"
    cv2.imwrite(processed_path, img)
    
    return detections, processed_filename


# -------------------
# Load Places365 (Scene Classification)
# -------------------
scene_model_file = "resnet18_places365.pth.tar"
scene_classes_file = "categories_places365.txt"

# Download model weights if not found
if not os.path.exists(scene_model_file):
    print("Downloading Places365 model...")
    import urllib.request
    url = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
    urllib.request.urlretrieve(url, scene_model_file)

if not os.path.exists(scene_classes_file):
    print("Downloading Places365 categories...")
    import urllib.request
    url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
    urllib.request.urlretrieve(url, scene_classes_file)

# Load categories
with open(scene_classes_file) as f:
    scene_classes = [line.strip().split(" ")[0][3:] for line in f]

# Load model
scene_model = models.resnet18(num_classes=365)
checkpoint = torch.load(scene_model_file, map_location=torch.device("cpu"))
state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
scene_model.load_state_dict(state_dict)
scene_model.eval()

# Transform
scene_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def predict_scene(image_bytes, model, classes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = scene_tf(Image.fromarray(img_rgb)).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    pred = torch.argmax(logits, 1).item()

    return classes[pred]


# -------------------
# Load GPT-2 (Text Generation for Scene Descriptions)
# -------------------
print("Loading GPT-2 model for scene descriptions...")
try:
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()
    
    # Set pad token to eos token for GPT-2
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    
    print("✅ GPT-2 model loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load GPT-2 model: {e}")
    gpt2_tokenizer = None
    gpt2_model = None

def generate_text_description(scene_class, objects_detected):
    """Generate scene description using GPT-2 model with better prompting"""
    if gpt2_model is None or gpt2_tokenizer is None:
        return None
    
    try:
        # Prepare objects list
        object_names = [obj['label'] for obj in objects_detected]
        object_counts = {}
        for obj in object_names:
            object_counts[obj] = object_counts.get(obj, 0) + 1
        
        # Create readable object list
        object_list = []
        for obj, count in object_counts.items():
            if count == 1:
                object_list.append(f"a {obj}")
            else:
                object_list.append(f"{count} {obj}s")
        
        # Better prompt engineering for GPT-2
        scene_name = scene_class.replace('_', ' ')
        
# ... (previous code remains the same) ...

        if object_list:
            objects_str = ", ".join(object_list[:-1]) + ", and " + object_list[-1] if len(object_list) > 1 else object_list[0]
            # New prompt with role and style guidance
            prompt = f"""You are a descriptive narrator. In one concise and easy-to-understand sentence, describe the {scene_name}.
            Objects: {objects_str}.
            Description: This is a"""
        else:
            prompt = f"""You are a descriptive narrator. In one concise and easy-to-understand sentence, describe the {scene_name}.
            Description: This is a"""

        # Tokenize the prompt
        inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate with more conservative settings for better quality
        with torch.no_grad():
            outputs = gpt2_model.generate(
                inputs,
                max_length=inputs.shape[1] + 25,  # Shorter generation
                min_length=inputs.shape[1] + 8,   # Minimum 8 new tokens
                temperature=0.6,                  # Lower temperature for more coherent text
                do_sample=True,
                top_p=0.8,                       # More focused sampling
                repetition_penalty=1.2,          # Reduce repetition
                no_repeat_ngram_size=3,          # Prevent 3-gram repetition
                pad_token_id=gpt2_tokenizer.eos_token_id,
                early_stopping=True              # Stop at natural endings
            )
        
        # Decode the full output
        full_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the description part
        if "The environment appears" in full_text:
            # Keep the prompt part and clean completion
            description = full_text
        else:
            description = full_text
        
        # Clean up the output properly
        description = description.strip()
        
        # Stop at first sentence ending after the prompt
        prompt_end = len(prompt)
        remaining_text = description[prompt_end:]
        
        # Find first natural stopping point
        for ending in ['. ', '.\n', '.']:
            if ending in remaining_text:
                stop_idx = remaining_text.find(ending) + 1
                description = description[:prompt_end + stop_idx]
                break
        
        # Ensure proper sentence structure
        if not description.endswith('.'):
            description += '.'
        
        # Validate quality - reject if too short or has common GPT-2 artifacts
        bad_phrases = ['The following', 'These are', 'This is a list', 'Here are', 'The details about']
        if (len(description.strip()) < 30 or 
            any(phrase in description for phrase in bad_phrases) or
            description.count('.') > 3):
            return None
        
        # Remove the "Scene description: " prefix for final output
        if description.startswith("Scene description: "):
            description = description[19:]
        
        return description
        
    except Exception as e:
        print(f"Error generating GPT-2 description: {e}")
        return None
