import random

def format_response(scene, objects):
    return {
        "scene": scene,
        "objects_detected": objects
    }

def count_objects(objects_detected):
    """Helper function to count objects by type"""
    object_counts = {}
    for obj in objects_detected:
        label = obj['label']
        object_counts[label] = object_counts.get(label, 0) + 1
    return object_counts

def create_object_descriptions(object_counts):
    """Helper function to create readable object descriptions"""
    object_descriptions = []
    for label, count in object_counts.items():
        if count == 1:
            object_descriptions.append(f"a {label}")
        else:
            object_descriptions.append(f"{count} {label}s")
    return object_descriptions

def generate_scene_description(scene_class, objects_detected):
    """Generate a natural language description of the scene using GPT-2 with fallback"""
    
    # Try to use GPT-2 model first
    try:
        from app.models import generate_text_description
        gpt2_description = generate_text_description(scene_class, objects_detected)
        if gpt2_description and len(gpt2_description.strip()) > 20:
            print("✅ Using GPT-2 generated description")
            return gpt2_description
    except Exception as e:
        print(f"GPT-2 generation failed, using fallback: {e}")
    
    # Fallback to rule-based approach if GPT-2 fails
    print("📝 Using rule-based fallback description")
    return generate_fallback_description(scene_class, objects_detected)

def generate_fallback_description(scene_class, objects_detected):
    """Fallback rule-based description generation"""
    
    # Use helper functions to avoid code duplication
    object_counts = count_objects(objects_detected)
    object_descriptions = create_object_descriptions(object_counts)
    
    # Scene context templates based on scene class
    scene_templates = {
        # Traffic-related scenes
        'highway': [
            "This appears to be a busy highway scene",
            "This looks like a major roadway or highway",
            "This seems to be a highway or freeway environment"
        ],
        'street': [
            "This appears to be an urban street scene",
            "This looks like a city street",
            "This seems to be a typical street environment"
        ],
        'parking_lot': [
            "This appears to be a parking area",
            "This looks like a parking lot or garage",
            "This seems to be a vehicle parking space"
        ],
        'crosswalk': [
            "This appears to be a pedestrian crossing area",
            "This looks like a crosswalk or intersection",
            "This seems to be a pedestrian-friendly zone"
        ],
        'traffic_light': [
            "This appears to be a traffic-controlled intersection",
            "This looks like an area with traffic signals",
            "This seems to be a signalized intersection"
        ],
        # Indoor scenes
        'computer_room': [
            "This appears to be a computer room or office space",
            "This looks like a workspace with computer equipment",
            "This seems to be an office or computer lab environment"
        ],
        'office': [
            "This appears to be an office environment",
            "This looks like a workplace or business setting",
            "This seems to be a professional office space"
        ],
        'conference_room': [
            "This appears to be a conference or meeting room",
            "This looks like a business meeting space",
            "This seems to be a conference room environment"
        ],
        'classroom': [
            "This appears to be a classroom or educational space",
            "This looks like a learning environment",
            "This seems to be an educational setting"
        ],
        # Other common scenes
        'restaurant': [
            "This appears to be a restaurant or dining area",
            "This looks like a food service establishment",
            "This seems to be a dining environment"
        ],
        'kitchen': [
            "This appears to be a kitchen area",
            "This looks like a food preparation space",
            "This seems to be a kitchen environment"
        ],
        'bedroom': [
            "This appears to be a bedroom",
            "This looks like a sleeping area",
            "This seems to be a personal bedroom space"
        ],
        'living_room': [
            "This appears to be a living room",
            "This looks like a family room or lounge area",
            "This seems to be a comfortable living space"
        ]
    }
    
    # Find matching scene template
    scene_description = f"This appears to be a {scene_class.replace('_', ' ')}"
    for key, templates in scene_templates.items():
        if key in scene_class.lower():
            scene_description = random.choice(templates)
            break
    
    # Add object information
    if object_descriptions:
        if len(object_descriptions) == 1:
            scene_description += f" with {object_descriptions[0]}"
        elif len(object_descriptions) == 2:
            scene_description += f" with {object_descriptions[0]} and {object_descriptions[1]}"
        else:
            scene_description += f" with {', '.join(object_descriptions[:-1])}, and {object_descriptions[-1]}"
    
    # Add context based on detected objects and scene type
    vehicles = [obj for obj in objects_detected if obj['label'] in ['car', 'truck', 'bus', 'motorcycle']]
    people = [obj for obj in objects_detected if obj['label'] == 'person']
    traffic_items = [obj for obj in objects_detected if obj['label'] in ['traffic light', 'stop sign']]
    tech_items = [obj for obj in objects_detected if obj['label'] in ['laptop', 'tv', 'computer', 'keyboard', 'mouse']]
    furniture = [obj for obj in objects_detected if obj['label'] in ['chair', 'desk', 'table', 'couch', 'bed']]
    
    # Add contextual information based on scene type and objects
    if 'traffic' in scene_class or 'highway' in scene_class or 'street' in scene_class:
        if len(vehicles) >= 3:
            scene_description += ". The scene shows significant vehicular traffic"
        elif people and vehicles:
            scene_description += ". Both pedestrians and vehicles are present in the scene"
        elif traffic_items:
            scene_description += ". Traffic control infrastructure is visible"
    elif 'computer' in scene_class or 'office' in scene_class:
        if tech_items and furniture:
            scene_description += ". The workspace appears to be equipped for computer work"
        elif people and tech_items:
            scene_description += ". People are present in this technology-equipped environment"
    elif 'room' in scene_class or 'kitchen' in scene_class or 'bedroom' in scene_class:
        if people and furniture:
            scene_description += ". The space appears to be actively used"
        elif furniture:
            scene_description += ". The area is furnished for daily activities"
    
    scene_description += "."
    
    return scene_description
