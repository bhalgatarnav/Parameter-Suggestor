import requests
import base64
from PIL import Image
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, clipdrop_api_key: str):
        self.clipdrop_api_key = clipdrop_api_key
        self.relight_url = "https://clipdrop-api.co/image-relight/v1"
        
    def relight_image(self, image_data: bytes) -> Optional[Image.Image]:
        """
        Relight image using Clipdrop API
        """
        try:
            headers = {
                'x-api-key': self.clipdrop_api_key
            }
            
            files = {
                'image_file': ('image.jpg', image_data, 'image/jpeg'),
                'light_direction': (None, 'front'),  # Options: front, left, right, top
                'light_intensity': (None, '1.0')     # Range: 0.0 to 2.0
            }
            
            response = requests.post(
                self.relight_url,
                headers=headers,
                files=files
            )
            
            if response.status_code == 200:
                # Convert response to PIL Image
                return Image.open(io.BytesIO(response.content))
            else:
                logger.error(f"Clipdrop API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error in image relighting: {str(e)}")
            return None
            
    def process_image(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Main preprocessing pipeline
        """
        try:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Relight image
            relit_image = self.relight_image(img_byte_arr)
            if relit_image:
                return relit_image
            
            # If relighting fails, return original
            logger.warning("Relighting failed, using original image")
            return image
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return None 