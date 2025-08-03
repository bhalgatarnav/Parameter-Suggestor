import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import cv2
from dotenv import load_dotenv
import os

load_dotenv()

class ImageAnalyzer:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def calculate_roughness(self, image):
        """Calculate surface roughness using gradient analysis"""
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        return np.mean(gradient_magnitude)
    
    def detect_structural_pattern(self, image):
        """Detect repeating patterns using frequency analysis"""
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        return np.mean(magnitude_spectrum)
    
    def get_dominant_color(self, image):
        """Extract dominant color in HSV space"""
        hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        h, s = np.unravel_index(hist.argmax(), hist.shape)
        return (h, s, np.mean(hsv[:,:,2]))
    
    def extract_color_palette(self, image, n_colors=5):
        """Extract main color palette using k-means clustering"""
        pixels = np.float32(image).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return palette.tolist()
    
    def predict_metalness(self, image):
        """Estimate metallic properties using reflection analysis"""
        hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        saturation = hsv[:,:,1] / 255.0
        value = hsv[:,:,2] / 255.0
        return np.mean(saturation * value)
    
    def estimate_reflectivity(self, image):
        """Estimate surface reflectivity using highlight analysis"""
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        _, highlights = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return np.sum(highlights) / (gray.shape[0] * gray.shape[1])
    
    def get_clip_embedding(self, image):
        """Get CLIP embedding for the image"""
        inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
        image_features = self.clip_model.get_image_features(**inputs)
        return image_features.detach().numpy()
    
    def analyze_image(self, image_path):
        """Complete image analysis pipeline"""
        image = Image.open(image_path)
        image = image.convert('RGB')
        
        features = {
            'clip_embedding': self.get_clip_embedding(image),
            'texture': {
                'roughness': self.calculate_roughness(image),
                'pattern': self.detect_structural_pattern(image)
            },
            'color': {
                'dominant_hsv': self.get_dominant_color(image),
                'palette': self.extract_color_palette(image)
            },
            'material': {
                'metalness': self.predict_metalness(image),
                'reflectivity': self.estimate_reflectivity(image)
            }
        }
        
        return features 