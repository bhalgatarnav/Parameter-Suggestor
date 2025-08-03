from sqlalchemy import text
import numpy as np
from database import SessionLocal
from sklearn.metrics.pairwise import cosine_similarity

class MaterialMatcher:
    def __init__(self):
        self.db = SessionLocal()
        
    def __del__(self):
        self.db.close()
    
    def calculate_texture_similarity(self, input_features, db_features):
        """Calculate texture similarity score"""
        weights = {
            'roughness': 0.6,
            'pattern': 0.4
        }
        
        score = sum(
            weights[key] * (1 - abs(input_features[key] - db_features[key]))
            for key in weights
        )
        return score
    
    def calculate_color_similarity(self, input_palette, db_palette):
        """Calculate color palette similarity using cosine similarity"""
        return cosine_similarity(
            np.array(input_palette).reshape(1, -1),
            np.array(db_palette).reshape(1, -1)
        )[0][0]
    
    def calculate_material_similarity(self, input_features, db_features):
        """Calculate material property similarity"""
        weights = {
            'metalness': 0.3,
            'reflectivity': 0.7
        }
        
        score = sum(
            weights[key] * (1 - abs(input_features[key] - db_features[key]))
            for key in weights
        )
        return score
    
    def find_matches(self, features, intent, limit=3):
        """Find top matching materials based on features and intent"""
        inspiration_type = intent['inspiration_type']
        
        # Build dynamic query based on inspiration type
        query = """
        SELECT 
            m.id,
            m.name,
            m.preview_url,
            m.roughness,
            m.pattern,
            m.metalness,
            m.reflectivity,
            m.color_palette,
            m.tags
        FROM materials m
        """
        
        try:
            results = self.db.execute(text(query)).fetchall()
            
            # Calculate similarity scores
            scored_materials = []
            for result in results:
                score = 0
                
                if inspiration_type == 'texture':
                    score = self.calculate_texture_similarity(
                        features['texture'],
                        {'roughness': result.roughness, 'pattern': result.pattern}
                    )
                elif inspiration_type == 'color':
                    score = self.calculate_color_similarity(
                        features['color']['palette'],
                        result.color_palette
                    )
                else:  # material
                    score = self.calculate_material_similarity(
                        features['material'],
                        {'metalness': result.metalness, 'reflectivity': result.reflectivity}
                    )
                
                scored_materials.append({
                    'id': result.id,
                    'name': result.name,
                    'preview_url': result.preview_url,
                    'match_score': score,
                    'match_type': inspiration_type
                })
            
            # Sort by score and return top matches
            scored_materials.sort(key=lambda x: x['match_score'], reverse=True)
            return scored_materials[:limit]
            
        except Exception as e:
            print(f"Error finding matches: {str(e)}")
            return []
    
    def get_match_reasons(self, material, features, intent):
        """Generate explanation for why material was matched"""
        reasons = []
        
        if material['match_type'] == 'texture':
            similarity = self.calculate_texture_similarity(
                features['texture'],
                {'roughness': material['roughness'], 'pattern': material['pattern']}
            )
            if similarity > 0.8:
                reasons.append(f"Texture match: {similarity*100:.0f}%")
                
        elif material['match_type'] == 'color':
            similarity = self.calculate_color_similarity(
                features['color']['palette'],
                material['color_palette']
            )
            if similarity > 0.7:
                reasons.append("Color palette match")
                
        elif material['match_type'] == 'material':
            similarity = self.calculate_material_similarity(
                features['material'],
                {'metalness': material['metalness'], 'reflectivity': material['reflectivity']}
            )
            if similarity > 0.75:
                reasons.append(f"Material properties match: {similarity*100:.0f}%")
        
        return reasons 