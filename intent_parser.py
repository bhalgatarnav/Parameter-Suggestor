from openai import OpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv('API_KEY'))

class IntentParser:
    def __init__(self):
        self.inspiration_types = ["texture", "material", "color"]
        self.attribute_list = [
            "roughness", "pattern", "metalness", "reflectivity",
            "color_palette", "dominant_color"
        ]
        self.use_cases = [
            "furniture", "wallpaper", "flooring", "clothing",
            "accessories", "industrial", "architectural"
        ]
    
    def parse_user_intent(self, user_text, image_tags):
        """Parse user intent using GPT-4"""
        prompt = f"""
        Image shows: {', '.join(image_tags)}
        User says: "{user_text}"
        
        Based on the image tags and user text, determine the user's material inspiration intent.
        Output a JSON object with these fields:
        - inspiration_type: one of ["texture", "material", "color"]
        - priority_attributes: list of relevant attributes from ["roughness", "pattern", "metalness", "reflectivity", "color_palette", "dominant_color"]
        - use_case: one of ["furniture", "wallpaper", "flooring", "clothing", "accessories", "industrial", "architectural"]
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error parsing intent: {str(e)}")
            return self.get_default_intent()
    
    def get_default_intent(self):
        """Return default intent when parsing fails"""
        return {
            "inspiration_type": "material",
            "priority_attributes": ["roughness", "metalness"],
            "use_case": "furniture"
        }
    
    def detect_inspiration_type(self, features):
        """Auto-detect dominant inspiration type from image features"""
        scores = {
            'texture': features['texture']['roughness'] * 0.6 + 
                      features['texture']['pattern'] * 0.4,
            'color': sum(1 for color in features['color']['palette'] if color),
            'material': features['material']['metalness'] * 0.3 + 
                       features['material']['reflectivity'] * 0.7
        }
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def validate_intent(self, intent, features):
        """Validate and adjust intent based on image features"""
        if intent['inspiration_type'] == 'color' and \
           max(features['color']['dominant_hsv']) < 0.1:
            # Image too dark/monochrome for color matching
            return self.adjust_intent_for_monochrome(intent)
            
        return intent
    
    def adjust_intent_for_monochrome(self, intent):
        """Adjust intent for monochrome images"""
        intent['inspiration_type'] = 'texture'
        intent['priority_attributes'] = ['roughness', 'pattern']
        return intent 