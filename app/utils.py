import random

def format_response(scene, objects):
    return {
        "scene": scene,
        "objects_detected": objects
    }

def generate_scene_description(scene_class, objects_detected):
    """Generate a natural language description of the scene"""
    
    # Count objects by type
    object_counts = {}
    for obj in objects_detected:
        label = obj['label']
        object_counts[label] = object_counts.get(label, 0) + 1
    
    # Create object description
    object_descriptions = []
    for label, count in object_counts.items():
        if count == 1:
            object_descriptions.append(f"a {label}")
        else:
            object_descriptions.append(f"{count} {label}s")
    
    # Scene context templates based on scene class
    scene_templates = {
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
        ]
    }
    
    # Find matching scene template
    scene_description = "This appears to be a traffic-related scene"
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
    
    # Add context based on detected objects
    vehicles = [obj for obj in objects_detected if obj['label'] in ['car', 'truck', 'bus', 'motorcycle']]
    people = [obj for obj in objects_detected if obj['label'] == 'person']
    traffic_items = [obj for obj in objects_detected if obj['label'] in ['traffic light', 'stop sign']]
    
    if len(vehicles) >= 3:
        scene_description += ". The scene shows significant vehicular traffic"
    elif people and vehicles:
        scene_description += ". Both pedestrians and vehicles are present in the scene"
    elif traffic_items:
        scene_description += ". Traffic control infrastructure is visible"
    
    scene_description += "."
    
    return scene_description
