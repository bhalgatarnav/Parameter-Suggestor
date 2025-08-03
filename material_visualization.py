import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple
import logging
import os
import json

logger = logging.getLogger(__name__)

class MaterialVisualizer:
    def __init__(self):
        """Initialize the material visualization system"""
        self.logger = logger
        self.texture_cache = {}
        
        # Default material settings
        self.default_settings = {
            "resolution": (1024, 1024),
            "format": "png",
            "quality": 95,
            "mipmap_levels": 8
        }
    
    def generate_preview(
        self,
        material_properties: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate material preview and texture maps"""
        try:
            # Merge settings with defaults
            settings = {**self.default_settings, **(settings or {})}
            
            # Generate base texture maps
            texture_maps = self._generate_texture_maps(material_properties, settings)
            
            # Generate preview render
            preview = self._generate_preview_render(material_properties, texture_maps)
            
            return {
                "preview": preview,
                "texture_maps": texture_maps,
                "settings": settings
            }
            
        except Exception as e:
            self.logger.error(f"Error generating material preview: {str(e)}")
            return self._generate_fallback_preview()
    
    def _generate_texture_maps(
        self,
        properties: Dict[str, Any],
        settings: Dict[str, Any]
    ) -> Dict[str, Image.Image]:
        """Generate all necessary texture maps for the material"""
        try:
            resolution = settings["resolution"]
            
            # Generate base color map
            albedo = self._generate_albedo_map(properties, resolution)
            
            # Generate roughness map
            roughness = self._generate_roughness_map(properties, resolution)
            
            # Generate normal map
            normal = self._generate_normal_map(properties, resolution)
            
            # Generate metallic map
            metallic = self._generate_metallic_map(properties, resolution)
            
            # Generate ambient occlusion map
            ao = self._generate_ao_map(properties, resolution)
            
            return {
                "albedo": albedo,
                "roughness": roughness,
                "normal": normal,
                "metallic": metallic,
                "ao": ao
            }
            
        except Exception as e:
            self.logger.error(f"Error generating texture maps: {str(e)}")
            return self._generate_fallback_textures()
    
    def _generate_albedo_map(
        self,
        properties: Dict[str, Any],
        resolution: Tuple[int, int]
    ) -> Image.Image:
        """Generate albedo (base color) map"""
        # For now, return a solid color based on material properties
        color = properties.get("color", {"r": 0.5, "g": 0.5, "b": 0.5})
        rgb = (
            int(color["r"] * 255),
            int(color["g"] * 255),
            int(color["b"] * 255)
        )
        return Image.new('RGB', resolution, rgb)
    
    def _generate_roughness_map(
        self,
        properties: Dict[str, Any],
        resolution: Tuple[int, int]
    ) -> Image.Image:
        """Generate roughness map"""
        roughness = properties.get("roughness", 0.5)
        value = int(roughness * 255)
        return Image.new('L', resolution, value)
    
    def _generate_normal_map(
        self,
        properties: Dict[str, Any],
        resolution: Tuple[int, int]
    ) -> Image.Image:
        """Generate normal map"""
        # Default normal map (flat surface)
        return Image.new('RGB', resolution, (128, 128, 255))
    
    def _generate_metallic_map(
        self,
        properties: Dict[str, Any],
        resolution: Tuple[int, int]
    ) -> Image.Image:
        """Generate metallic map"""
        metalness = properties.get("metalness", 0.0)
        value = int(metalness * 255)
        return Image.new('L', resolution, value)
    
    def _generate_ao_map(
        self,
        properties: Dict[str, Any],
        resolution: Tuple[int, int]
    ) -> Image.Image:
        """Generate ambient occlusion map"""
        # Default AO map (fully lit)
        return Image.new('L', resolution, 255)
    
    def _generate_preview_render(
        self,
        properties: Dict[str, Any],
        texture_maps: Dict[str, Image.Image]
    ) -> Image.Image:
        """Generate a preview render of the material"""
        # For now, return the albedo map as preview
        return texture_maps["albedo"]
    
    def _generate_fallback_preview(self) -> Dict[str, Any]:
        """Generate fallback preview when generation fails"""
        resolution = self.default_settings["resolution"]
        
        # Create default textures
        fallback_textures = {
            "albedo": Image.new('RGB', resolution, (128, 128, 128)),
            "roughness": Image.new('L', resolution, 128),
            "normal": Image.new('RGB', resolution, (128, 128, 255)),
            "metallic": Image.new('L', resolution, 0),
            "ao": Image.new('L', resolution, 255)
        }
        
        return {
            "preview": fallback_textures["albedo"],
            "texture_maps": fallback_textures,
            "settings": self.default_settings
        }
    
    def _generate_fallback_textures(self) -> Dict[str, Image.Image]:
        """Generate fallback textures when generation fails"""
        resolution = self.default_settings["resolution"]
        return {
            "albedo": Image.new('RGB', resolution, (128, 128, 128)),
            "roughness": Image.new('L', resolution, 128),
            "normal": Image.new('RGB', resolution, (128, 128, 255)),
            "metallic": Image.new('L', resolution, 0),
            "ao": Image.new('L', resolution, 255)
        } 