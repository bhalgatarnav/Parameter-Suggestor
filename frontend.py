import streamlit as st
import requests
import base64
import json
from PIL import Image
import io
import time
from typing import Optional
import os
from dotenv import load_dotenv
from utils.network_generator import generate_material_network, process_network_data, calculate_node_positions
from utils.network_visualizer import create_network_visualization, render_network, show_material_card
from database import SessionLocal

# Load environment variables
load_dotenv()

# Get backend URL from environment variables or use default
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

st.set_page_config(
    layout="wide", 
    page_title="Material AI Advisor", 
    page_icon="🧊",
    initial_sidebar_state="collapsed"
)

# Add status indicator for backend connection
def check_backend_health():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Display backend status in sidebar
with st.sidebar:
    st.title("System Status")
    if check_backend_health():
        st.success("Backend Connected")
        st.write(f"Backend URL: {BACKEND_URL}")
    else:
        st.error("Backend Not Connected")
        st.write(f"Attempted Backend URL: {BACKEND_URL}")
        st.write("Please check if:")
        st.write("1. Backend server is running")
        st.write("2. BACKEND_URL in .env is correct")
        st.write("3. No firewall blocking the connection")

# Add Three.js dependencies and custom viewer component
st.markdown("""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/RGBELoader.js"></script>
    <script>
    class MaterialViewer {
        constructor(containerId, width, height) {
            this.container = document.getElementById(containerId);
            this.scene = new THREE.Scene();
            this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
            this.renderer = new THREE.WebGLRenderer({ 
                antialias: true,
                alpha: true 
            });
            this.controls = null;
            this.currentMesh = null;
            this.envMap = null;
            this.lights = {};
            
            // Available geometries
            this.geometries = {
                sphere: new THREE.SphereGeometry(2, 64, 64),
                cube: new THREE.BoxGeometry(2, 2, 2),
                cylinder: new THREE.CylinderGeometry(1, 1, 3, 32),
                torus: new THREE.TorusGeometry(1.5, 0.5, 32, 100),
                cone: new THREE.ConeGeometry(1.5, 3, 32)
            };
            
            this.init(width, height);
        }
        
        init(width, height) {
            // Setup renderer with physically correct lighting
            this.renderer.setSize(width, height);
            this.renderer.setPixelRatio(window.devicePixelRatio);
            this.renderer.physicallyCorrectLights = true;
            this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
            this.renderer.toneMappingExposure = 1;
            this.renderer.outputEncoding = THREE.sRGBEncoding;
            this.container.appendChild(this.renderer.domElement);
            
            // Setup camera
            this.camera.position.z = 5;
            
            // Setup controls
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            
            // Initialize lighting presets
            this.setupLightingPresets();
            
            // Load default environment map
            this.loadEnvironmentMap('studio');
            
            // Add default geometry
            this.changeGeometry('sphere');
            
            // Start animation loop
            this.animate();
        }
        
        setupLightingPresets() {
            // Studio lighting preset
            this.lights.studio = {
                ambient: new THREE.AmbientLight(0xffffff, 0.5),
                key: new THREE.DirectionalLight(0xffffff, 1.5),
                fill: new THREE.DirectionalLight(0xffffff, 0.5),
                back: new THREE.DirectionalLight(0xffffff, 0.5)
            };
            
            // Outdoor lighting preset
            this.lights.outdoor = {
                ambient: new THREE.AmbientLight(0x90c0ff, 0.8),
                sun: new THREE.DirectionalLight(0xffffff, 2.0),
                sky: new THREE.HemisphereLight(0x90c0ff, 0x804040, 0.5)
            };
            
            // Dramatic lighting preset
            this.lights.dramatic = {
                ambient: new THREE.AmbientLight(0x202020, 0.3),
                spot1: new THREE.SpotLight(0xffffff, 1.0),
                spot2: new THREE.SpotLight(0xff4040, 0.5)
            };
            
            // Product lighting preset
            this.lights.product = {
                ambient: new THREE.AmbientLight(0xffffff, 0.4),
                key: new THREE.DirectionalLight(0xffffff, 1.0),
                fill: new THREE.DirectionalLight(0xffffff, 0.4),
                rim: new THREE.DirectionalLight(0xffffff, 0.8)
            };
        }
        
        setLightingPreset(preset) {
            // Clear existing lights
            this.scene.children
                .filter(child => child instanceof THREE.Light)
                .forEach(light => this.scene.remove(light));
            
            // Apply new preset
            const lights = this.lights[preset];
            if (lights) {
                Object.values(lights).forEach(light => {
                    if (light instanceof THREE.DirectionalLight) {
                        light.castShadow = true;
                    }
                    if (preset === 'dramatic' && light instanceof THREE.SpotLight) {
                        light.position.set(Math.random() * 10 - 5, 5, 5);
                        light.lookAt(0, 0, 0);
                    } else if (light instanceof THREE.DirectionalLight) {
                        light.position.set(5, 5, 5);
                    }
                    this.scene.add(light);
                });
            }
        }
        
        async loadEnvironmentMap(preset) {
            const loader = new THREE.RGBELoader();
            const envMaps = {
                studio: 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/equirectangular/venice_sunset_1k.hdr',
                outdoor: 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/equirectangular/pedestrian_overpass_1k.hdr',
                night: 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/equirectangular/moonless_golf_1k.hdr'
            };
            
            try {
                const texture = await loader.loadAsync(envMaps[preset] || envMaps.studio);
                texture.mapping = THREE.EquirectangularReflectionMapping;
                this.scene.environment = texture;
                this.envMap = texture;
                
                if (this.currentMesh && this.currentMesh.material) {
                    this.currentMesh.material.envMap = texture;
                    this.currentMesh.material.needsUpdate = true;
                }
            } catch (error) {
                console.error('Error loading environment map:', error);
            }
        }
        
        changeGeometry(geometryType) {
            const geometry = this.geometries[geometryType];
            if (!geometry) return;
            
            if (this.currentMesh) {
                this.scene.remove(this.currentMesh);
            }
            
            const material = new THREE.MeshStandardMaterial({
                color: 0x808080,
                roughness: 0.5,
                metalness: 0.5,
                envMap: this.envMap,
                normalScale: new THREE.Vector2(1, 1),
                envMapIntensity: 1.0
            });
            
            this.currentMesh = new THREE.Mesh(geometry, material);
            this.currentMesh.castShadow = true;
            this.currentMesh.receiveShadow = true;
            this.scene.add(this.currentMesh);
        }
        
        async loadCustomModel(url) {
            const loader = new THREE.GLTFLoader();
            try {
                const gltf = await loader.loadAsync(url);
                if (this.currentMesh) {
                    this.scene.remove(this.currentMesh);
                }
                this.currentMesh = gltf.scene;
                this.scene.add(this.currentMesh);
                
                // Apply material to all meshes in the model
                this.currentMesh.traverse((child) => {
                    if (child instanceof THREE.Mesh) {
                        child.material = new THREE.MeshStandardMaterial({
                            color: 0x808080,
                            roughness: 0.5,
                            metalness: 0.5,
                            envMap: this.envMap,
                            normalScale: new THREE.Vector2(1, 1),
                            envMapIntensity: 1.0
                        });
                    }
                });
            } catch (error) {
                console.error('Error loading custom model:', error);
            }
        }
        
        updateMaterial(properties) {
            if (!this.currentMesh) return;
            
            const updateMesh = (mesh) => {
                if (mesh.material) {
                    // Update basic properties
                    mesh.material.roughness = properties.roughness || 0.5;
                    mesh.material.metalness = properties.metalness || 0.5;
                    if (properties.color) {
                        mesh.material.color.setHex(properties.color);
                    }
                    
                    // Update normal map
                    if (properties.normalMap) {
                        const normalTexture = new THREE.TextureLoader().load(properties.normalMap);
                        mesh.material.normalMap = normalTexture;
                        mesh.material.normalScale.set(
                            properties.normalScale || 1.0,
                            properties.normalScale || 1.0
                        );
                    }
                    
                    // Update displacement
                    if (properties.displacementMap) {
                        const dispTexture = new THREE.TextureLoader().load(properties.displacementMap);
                        mesh.material.displacementMap = dispTexture;
                        mesh.material.displacementScale = properties.displacementScale || 1.0;
                    }
                    
                    // Update environment map intensity
                    if (properties.envMapIntensity !== undefined) {
                        mesh.material.envMapIntensity = properties.envMapIntensity;
                    }
                    
                    mesh.material.needsUpdate = true;
                }
            };
            
            if (this.currentMesh.traverse) {
                this.currentMesh.traverse((child) => {
                    if (child instanceof THREE.Mesh) {
                        updateMesh(child);
                    }
                });
            } else {
                updateMesh(this.currentMesh);
            }
        }
        
        animate() {
            requestAnimationFrame(() => this.animate());
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    // Initialize viewers for each material card
    function initMaterialViewers() {
        const previewContainers = document.querySelectorAll('.preview-container');
        previewContainers.forEach((container, index) => {
            const viewer = new MaterialViewer(`material-viewer-${index}`, 600, 300);
            window[`materialViewer${index}`] = viewer;
        });
    }
    
    // Call after Streamlit is loaded
    if (window.Streamlit) {
        window.Streamlit.addEventListener('load', initMaterialViewers);
    }
    </script>
""", unsafe_allow_html=True)

# Custom CSS - Improved contrast and readability
st.markdown("""
<style>
:root {
    --primary: #007bff;  /* Standard web UI blue */
    --primary-hover: #0056b3;  /* Darker blue for hover */
    --text-color: #000000;  /* Black text */
    --background: #ffffff;  /* White background */
    --border-color: #e0e0e0;
}

/* Base layout */
body {
    background: var(--background);
    color: var(--text-color);
    font-family: 'Inter', system-ui, sans-serif;
}

.stApp {
    max-width: 1600px;
    padding: 1.5rem;
    margin: auto;
    background: var(--background);
}

/* Header */
.header {
    background: var(--background);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 6px 25px rgba(0,0,0,0.1);
    color: var(--text-color);
    text-align: left;
    border: 1px solid var(--border-color);
}

/* Material name styling - increased size */
.stExpander > div[data-testid="stExpander"] > div[role="button"] {
    font-size: 1.5rem !important;  /* Increased from 1.25rem to 1.5rem */
    font-weight: 600 !important;
    color: var(--text-color) !important;
    margin-bottom: 0.5rem !important;
}

/* Properties and Sustainability boxes */
.property-box {
    background: var(--background) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
}

/* Material Preview section */
.material-preview {
    width: 100%;
    height: 400px;
    background: var(--background);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    margin-bottom: 1rem;
    position: relative;
}

.material-preview canvas {
    width: 100%;
    height: 100%;
}

/* Text alignment improvements */
.property-label {
    font-weight: 500;
    margin-bottom: 0.25rem;
    display: block;
}

.property-value {
    margin-bottom: 1rem;
    display: block;
}

/* Quick action buttons alignment */
.quick-actions {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

/* Resolution selector alignment */
.resolution-selector {
    display: flex;
    gap: 0.25rem;
    margin-bottom: 1rem;
}

.resolution-btn {
    padding: 0.25rem 0.75rem;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    background: var(--background);
    color: var(--text-color);
    cursor: pointer;
}

.resolution-btn.active {
    background: var(--primary);
    color: white;
    border-color: var(--primary);
}

/* Control groups alignment */
.control-group {
    margin-bottom: 1.5rem;
    padding: 1rem;
    border: 1px solid var(--border-color);
    border-radius: 4px;
}

.control-group h4 {
    margin-top: 0;
    margin-bottom: 0.75rem;
    font-size: 1rem;
    font-weight: 500;
}

/* Material channels grid */
.channel-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.channel-tile {
    aspect-ratio: 1;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}

.channel-tile img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.channel-tile h5 {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    margin: 0;
    padding: 0.5rem;
    background: rgba(255, 255, 255, 0.9);
    text-align: center;
}

/* Material card */
.stExpander {
    background: var(--background) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    margin-bottom: 1rem !important;
    color: var(--text-color) !important;
}

/* Expander content */
.stExpander > div {
    background: var(--background) !important;
    color: var(--text-color) !important;
}

/* Property values and labels */
.property-value,
.property-label,
.stMarkdown p,
.stMarkdown div,
.stMarkdown span {
    color: var(--text-color) !important;
}

/* Quick action buttons */
.quick-action-btn {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    transition: background-color 0.3s !important;
}

.quick-action-btn:hover {
    background: var(--primary-hover) !important;
}

/* Section titles */
.section-title {
    color: var(--text-color) !important;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color);
}

/* Material preview container */
    .preview-container {
    background: var(--background);
    border: 1px solid var(--border-color);
}

/* Material controls */
.material-controls {
    background: var(--background);
    border: 1px solid var(--border-color);
}

.control-group {
    background: var(--background);
    color: var(--text-color);
}

.control-group h4 {
    color: var(--text-color) !important;
}

/* Badges */
.badge {
    background: var(--primary);
    color: white;
}

/* Empty state */
.empty-state {
    color: var(--text-color);
}

.empty-state h3 {
    color: var(--text-color) !important;
    font-size: 1.25rem !important;
}

.empty-state p {
    color: var(--text-color) !important;
}

/* Footer */
.footer {
    color: var(--text-color) !important;
}

/* Radio buttons and checkboxes */
.stRadio > div > div > label,
.stCheckbox > div > div > label {
    color: var(--text-color) !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    color: var(--text-color) !important;
}

/* Markdown text */
.stMarkdown {
    color: var(--text-color) !important;
}

/* AI Reasoning box */
.ai-reasoning {
    background: var(--background) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
}

/* Material map containers */
.map-container {
    width: 100%;
    max-width: 200px;  /* Limit maximum width */
    margin-bottom: 1rem;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    overflow: hidden;
}

.map-container img {
    width: 100%;
    height: auto;
    max-height: 150px;  /* Limit maximum height */
    object-fit: contain;
    display: block;
}

.map-container .map-label {
    padding: 0.5rem;
    text-align: center;
    border-top: 1px solid var(--border-color);
    background: var(--background);
    color: var(--text-color);
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)

# Custom CSS - Material map sizing
st.markdown("""
<style>
/* Material map image sizing */
img[alt$="Map"] {
    width: 200px !important;
    height: 150px !important;
    object-fit: contain !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 4px !important;
    margin-bottom: 8px !important;
}

/* Map label styling */
div:has(> img[alt$="Map"]) + div {
    text-align: center !important;
    font-size: 14px !important;
    margin-bottom: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# Update the material maps rendering
def render_material_maps(material: dict):
    st.markdown('<div class="material-maps">', unsafe_allow_html=True)
    for map_type in ["Albedo", "Roughness", "Normal", "Specular"]:
        st.markdown(f"""
        <div class="map-container">
            <img src="{material.get('preview_image', 'https://via.placeholder.com/150x100.png')}?text={map_type}" 
                 alt="{map_type} Map">
            <div class="map-label">{map_type}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'query': "",
        'results': [],
        'uploaded_img': None,
        'selected_material': None,
        'refined_query': "",
        'selected_nodes': set(),
        'property_settings': {
            "roughness": 0.5,
            "metalness": 0.5,
            "opacity": 1.0,
            "normal_strength": 0.8,
            "material_type": "Any",
            "wood_grain": "linear",
            "wet_look": False,
            "sustainability": 3
        },
        'feedback_submitted': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Initialize material database connection
def get_material_db():
    if 'material_db' not in st.session_state:
        st.session_state.material_db = SessionLocal()
    return st.session_state.material_db

# Header section
def render_header():
    """Render the application header"""
    st.markdown("""
    <div class="header">
        <h1>🧊 Material AI Advisor</h1>
        <p>Discover, create, and refine PBR materials with AI-powered recommendations</p>
    </div>
    """, unsafe_allow_html=True)

# Visual nodes configuration
VISUAL_NODES = {
    "Material Types": ["Metal", "Wood", "Plastic", "Fabric", "Glass", "Ceramic"],
    "Surface Qualities": ["Rough", "Glossy", "Textured", "Smooth", "Porous", "Matte"],
    "Conditions": ["Worn", "New", "Weathered", "Polished", "Damaged", "Natural"],
    "Effects": ["Wet", "Frosted", "Translucent", "Metallic", "Anisotropic", "Iridescent"]
}

def handle_visual_node_selection(category: str, selected: list):
    """Handle dropdown selection changes"""
    # Clear existing nodes from this category
    st.session_state.selected_nodes = {n for n in st.session_state.selected_nodes 
                                     if n not in VISUAL_NODES[category]}
    
    # Add newly selected nodes
    st.session_state.selected_nodes.update(selected)
    
    # Update query
    st.session_state.query = " ".join(st.session_state.selected_nodes) if st.session_state.selected_nodes else ""

def render_visual_nodes():
    """Render dropdown-based visual node selection with network visualization"""
    st.markdown("Select material characteristics:")
    
    # Get currently selected nodes from each category
    selections = {}
    for category, nodes in VISUAL_NODES.items():
        current_selection = [n for n in st.session_state.selected_nodes if n in nodes]
        
        # Create multi-select dropdown
        selected = st.multiselect(
            label=f"**{category}**",
            options=nodes,
            default=current_selection,
            key=f"dropdown_{category.lower().replace(' ', '_')}"
        )
        
        selections[category.lower().replace(' ', '_')] = selected
        
        # Update selection state
        if selected != current_selection:
            handle_visual_node_selection(category, selected)
    
    # Only show network if selections are made
    if any(selections.values()):
        st.markdown("### Material Relationship Network")
        
        # Generate network data
        network_data = generate_material_network(selections)
        network_data = process_network_data(network_data, get_material_db())
        network_data = calculate_node_positions(network_data)
        
        # Create and render network
        html_file = create_network_visualization(network_data)
        render_network(html_file)
        
        # Handle clicked node
        if "clicked_material" in st.session_state:
            show_material_card(st.session_state.clicked_material)
    else:
        st.info("Select material characteristics to see relationships")

# Add JavaScript for node click handling
st.markdown("""
<script>
function handleNodeClick(node) {
    fetch('/_stcore/handle_node_click', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({node: node})
    }).then(() => window.location.reload());
}
</script>
""", unsafe_allow_html=True)

# Material property controls
def render_property_controls():
    """Render material property sliders and controls"""
    with st.container():
        st.markdown('<div class="section-title">Material Properties</div>', unsafe_allow_html=True)
        
        col_sliders = st.columns(2)
        with col_sliders[0]:
            st.session_state.property_settings["roughness"] = st.slider(
                "Roughness", 0.0, 1.0, 
                st.session_state.property_settings["roughness"], 0.01,
                key="slider_roughness",
                help="Surface microdetail that scatters light (0 = smooth, 1 = rough)"
            )
            st.session_state.property_settings["metalness"] = st.slider(
                "Metalness", 0.0, 1.0, 
                st.session_state.property_settings["metalness"], 0.01,
                key="slider_metalness",
                help="Metallic appearance (0 = dielectric, 1 = metallic)"
            )
        with col_sliders[1]:
            st.session_state.property_settings["opacity"] = st.slider(
                "Opacity", 0.0, 1.0, 
                st.session_state.property_settings["opacity"], 0.01,
                key="slider_opacity",
                help="Material transparency (0 = transparent, 1 = opaque)"
            )
            st.session_state.property_settings["normal_strength"] = st.slider(
                "Normal Strength", 0.0, 1.0, 
                st.session_state.property_settings["normal_strength"], 0.01,
                key="slider_normal_strength",
                help="Intensity of surface details from normal map"
            )
    
    # Semantic controls
    st.markdown('<div class="section-title">Semantic Controls</div>', unsafe_allow_html=True)
    with st.container():
        col_semantic = st.columns(2)
        with col_semantic[0]:
            st.session_state.property_settings["material_type"] = st.selectbox(
                "Material Type", 
                ["Any", "Metal", "Wood", "Plastic", "Fabric", "Glass", "Stone", "Ceramic", "Composite"],
                key="select_material_type"
            )
            
            if st.session_state.property_settings["material_type"] == "Wood":
                st.session_state.property_settings["wood_grain"] = st.selectbox(
                    "Wood Grain", 
                    ["Linear", "Radial", "Wavy", "Knotty"],
                    key="select_wood_grain"
                )
                
        with col_semantic[1]:
            st.session_state.property_settings["sustainability"] = st.slider(
                "Sustainability Preference", 1, 5, 
                st.session_state.property_settings["sustainability"],
                key="slider_sustainability",
                help="1 = Low priority, 5 = High priority"
            )
            st.session_state.property_settings["wet_look"] = st.checkbox(
                "Wet Look Effect", 
                st.session_state.property_settings["wet_look"],
                key="checkbox_wet_look"
            )

# Main application layout
def main():
    render_header()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Visual Intent Expression
        st.markdown('<div class="section-title">Visual Intent Expression</div>', unsafe_allow_html=True)
        visual_option = st.radio(
            "Express your material visually:",
            ("Upload Image", "Visual Nodes", "Text Prompt"),
            horizontal=True,
            key="visual_option"
        )
        
        if visual_option == "Upload Image":
            handle_image_upload()
        elif visual_option == "Visual Nodes":
            render_visual_nodes()
        else:  # Text Prompt
            st.session_state.query = st.text_area(
                "Describe your material in natural language:",
                value=st.session_state.query,
                height=120,
                placeholder="e.g., 'rusty metal under wet conditions' or 'frosted glass with light diffusion'"
            )
        
        # Material properties
        render_property_controls()
        
        # Refinement
        st.markdown('<div class="section-title">Refine Your Prompt</div>', unsafe_allow_html=True)
        st.session_state.refined_query = st.text_input(
            "Add descriptive words to refine your material:",
            value=st.session_state.refined_query,
            placeholder="e.g., 'more reflective', 'less porous', 'warmer tones'"
        )
        
        # Generate button
        if st.button("Discover Materials", type="primary", use_container_width=True, key="btn_generate_main"):
            if not st.session_state.query and not st.session_state.uploaded_img:
                st.warning("Please describe your material or upload an image")
            else:
                generate_recommendations()
    
    with col2:
        # Results display
        if st.session_state.results:
            st.markdown(f'<div class="section-title">Recommended Materials ({len(st.session_state.results)})</div>', 
                       unsafe_allow_html=True)
            
            for idx, material in enumerate(st.session_state.results):
                render_material_card(material, idx)
        else:
            # Empty state
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🧊</div>
                <h3>Discover Your Perfect Material</h3>
                <p>Describe your material needs using text, images, or visual nodes</p>
                <p>Try examples: "weathered copper", "translucent marble", "matte black plastic"</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; color: #718096;">
        <p>Material AI Advisor v1.1 • Integrated with KeyShot • Powered by GPT-4 and Material Science AI</p>
    </div>
    """, unsafe_allow_html=True)

def handle_image_upload():
    """Handle image upload and analysis"""
    uploaded_file = st.file_uploader(
        "Upload a photo of the material you want to replicate",
        type=["jpg", "png", "jpeg"],
        key="uploader_material_image"
    )
    
    if uploaded_file:
        try:
            # Convert to PIL Image
            image = Image.open(uploaded_file)
            
            # Display original image and context inputs first
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Add context inputs
                st.markdown("### Image Context")
                image_content = st.text_input(
                    "What does this image show? (e.g., tree bark, metal surface, fabric)",
                    key="image_content_upload"
                )
                
                inspiration_type = st.selectbox(
                    "What aspect are you interested in?",
                    ["Texture", "Material", "Color", "Finish"],
                    key="inspiration_type_upload"
                )
                
                additional_notes = st.text_area(
                    "Any additional details about what you're looking for?",
                    placeholder="e.g., I want to find materials with similar roughness but in metallic finish",
                    key="additional_notes_upload"
                )
                
                # Store context in session state
                if image_content:
                    st.session_state.current_image_context = {
                        "content": image_content,
                        "inspiration_type": inspiration_type,
                        "additional_notes": additional_notes
                    }
                
                analyze_button = st.button("Analyze Image", key="analyze_button_upload")
            
            if analyze_button and image_content:
                with st.spinner("Analyzing material properties..."):
                    try:
                        # Convert image to bytes
                        buffered = io.BytesIO()
                        image.save(buffered, format="JPEG")
                        
                        # Prepare file upload with context
                        files = {
                            "file": ("image.jpg", buffered.getvalue(), "image/jpeg")
                        }
                        
                        # Add context to form data
                        data = {
                            "image_content": image_content,
                            "inspiration_type": inspiration_type,
                            "additional_notes": additional_notes
                        }
                        
                        response = requests.post(
                            f"{BACKEND_URL}/analyze-image",
                            files=files,
                            data=data,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            analysis = response.json()
                            
                            # Display analysis results
                            with col2:
                                st.markdown("### Material Analysis")
                                
                                if analysis.get("success", False):
                                    data = analysis.get("data", {})
                                    
                                    # Primary material
                                    material_analysis = data.get("material_analysis", {})
                                    st.markdown(f"""
                                        **Primary Material**: {material_analysis.get('primary_material', 'Unknown')}  
                                        **Confidence**: {material_analysis.get('confidence', 0):.2%}
                                    """)
                                    
                                    # Secondary materials
                                    st.markdown("**Secondary Materials**:")
                                    for mat in material_analysis.get('secondary_materials', []):
                                        st.markdown(f"- {mat['material']} ({mat['confidence']:.2%})")
                                    
                                    # Surface properties
                                    st.markdown("### Surface Properties")
                                    props = data.get('surface_properties', {})
                                    st.markdown(f"""
                                        - Roughness: {props.get('roughness', 0):.2f}
                                        - Metalness: {props.get('metalness', 0):.2f}
                                        - Normal Strength: {props.get('normal_strength', 0):.2f}
                                        - Displacement: {props.get('displacement_scale', 0):.2f}
                                    """)
                                    
                                    # Update query with material description
                                    material_desc = f"{material_analysis.get('primary_material', 'Unknown')} material with "
                                    material_desc += f"roughness {props.get('roughness', 0):.2f} and metalness {props.get('metalness', 0):.2f}"
                                    st.session_state.query = material_desc
                                else:
                                    error = analysis.get("error", {})
                                    st.error(f"Analysis failed: {error.get('message', 'Unknown error')}")
                        else:
                            st.error(f"Analysis failed: {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection error: {str(e)}")
                        st.info("Backend server appears to be offline. Please check if it's running.")
            elif analyze_button:
                st.warning("Please provide a description of what the image shows before analyzing.")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try another image or adjust the lighting conditions.")

def generate_recommendations():
    """Generate material recommendations based on query and properties"""
    if not st.session_state.query and not st.session_state.uploaded_img:
        st.warning("Please describe your material or upload an image")
        return
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    with st.spinner("Generating AI-powered recommendations..."):
        try:
            # First check if backend is alive
            health_check = requests.get(f"{BACKEND_URL}/health", timeout=2)
            if health_check.status_code != 200:
                st.error("Backend server is not responding")
                st.info("Please start the backend with: uvicorn main:app --reload")
                return
            
            # Make recommendation request
            response = requests.post(
                f"{BACKEND_URL}/recommend",
                headers=headers,
                json={
                    "query": st.session_state.query,
                    "properties": st.session_state.property_settings
                },
                timeout=20
            )
            
            if response.status_code == 403:
                st.error("Permission denied. Please check backend CORS settings.")
                st.code(f"Backend URL: {BACKEND_URL}", language="bash")
                return
        
            if response.status_code == 200:
                st.session_state.results = response.json().get("materials", [])
                st.success(f"Found {len(st.session_state.results)} materials!")
            else:
                st.error(f"Backend error: {response.text}")
        
        except requests.exceptions.ConnectionError:
            st.error("Backend server is not running")
            st.info("Start your backend with: uvicorn main:app --reload")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.info("Please check your backend configuration and try again")

# Material card rendering
def render_material_card(material: dict, idx: int):
    """Render a single material recommendation card"""
    with st.expander(f"### {material['name']}", expanded=True):
        # Get current context from session state
        current_context = st.session_state.get('current_image_context', {})
        inspiration_type = current_context.get('inspiration_type', '')
        
        # Material header
        st.caption(f"KeyShot Path: {material['key_shot_path']}")
        st.progress(material.get("similarity_score", 0.7), text="Relevance Score")
        
        # AI reasoning with reference image context
        with st.expander("AI Reasoning", expanded=True):
            base_reasoning = material.get("reasoning", "No reasoning provided.")
            
            # Add reference image context if available
            if inspiration_type and current_context.get('content'):
                reference_context = f"\n\nAlignment with reference {current_context['content']}: "
                if inspiration_type == "Texture":
                    reference_context += "This material matches the textural qualities you're looking for, particularly in terms of surface pattern and roughness."
                elif inspiration_type == "Color":
                    reference_context += "The color palette aligns well with your reference image, capturing similar tones and variations."
                elif inspiration_type == "Material":
                    reference_context += "This material shares similar physical properties with your reference, while potentially offering improved performance characteristics."
                elif inspiration_type == "Finish":
                    reference_context += "The surface finish resembles your reference image, with comparable light interaction and tactile qualities."
                
                st.markdown(f'<div class="ai-reasoning">{base_reasoning}{reference_context}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-reasoning">{base_reasoning}</div>', 
                           unsafe_allow_html=True)
        
        # Rest of the material card rendering
        # Tags and metadata
        if any(key in material["properties"] for key in ["appearance_tags", "intent_tags", "keywords"]):
            with st.expander("Tags & Metadata", expanded=False):
                if material["properties"].get("appearance_tags"):
                    st.markdown("**Appearance Tags:**")
                    st.markdown(", ".join(material["properties"]["appearance_tags"]))
                
                if material["properties"].get("intent_tags"):
                    st.markdown("**Intent Tags:**")
                    st.markdown(", ".join(material["properties"]["intent_tags"]))
                
                if material["properties"].get("keywords"):
                    st.markdown("**Keywords:**")
                    st.markdown(", ".join(material["properties"]["keywords"]))
        
        # Use cases
        if material["properties"].get("use_cases"):
            with st.expander("Suggested Use Cases", expanded=False):
                for use_case in material["properties"]["use_cases"]:
                    st.markdown(f"- {use_case}")
        
        # Material Preview section
        st.subheader("Material Preview")
        
        # Quick Actions Bar
        st.markdown("""
<div class="quick-actions">
    <button class="quick-action-btn">📸 Screenshot</button>
    <button class="quick-action-btn">💾 Save</button>
    <button class="quick-action-btn">🔄 Reset</button>
    <button class="quick-action-btn">📤 Export</button>
</div>
""", unsafe_allow_html=True)
        
        # Resolution selector
        st.markdown("""
<div class="resolution-selector">
    <button class="resolution-btn">1K</button>
    <button class="resolution-btn active">2K</button>
    <button class="resolution-btn">4K</button>
    <button class="resolution-btn">8K</button>
</div>
""", unsafe_allow_html=True)
        
        preview_col1, preview_col2 = st.columns([3, 1])
        
        with preview_col1:
            # 3D Preview with Three.js
            st.markdown(f"""
            <div class="material-preview" id="material-viewer-{idx}">
                <!-- Three.js will render here -->
            </div>
            <script>
                if (window[`materialViewer{idx}`]) {{
                    window[`materialViewer{idx}`].updateMaterial({{
                        roughness: {material["properties"]["roughness"]},
                        metalness: {material["properties"]["metalness"]},
                        color: 0x{int(material["properties"]["color"]["r"]*255):02x}{int(material["properties"]["color"]["g"]*255):02x}{int(material["properties"]["color"]["b"]*255):02x},
                        normalScale: {material["properties"].get("normal_scale", 1.0)},
                        envMapIntensity: {material["properties"].get("env_map_intensity", 1.0)}
                    }});
                }}
            </script>
            """, unsafe_allow_html=True)
            
            # Material Controls
            with st.container():
                st.markdown('<div class="material-controls">', unsafe_allow_html=True)
                
                # Preview controls in a grid
                control_cols = st.columns(3)
                
                with control_cols[0]:
                    st.markdown("""
                    <div class="control-group">
                        <h4>Geometry</h4>
                    """, unsafe_allow_html=True)
                    geometry = st.selectbox(
                        "",
                        ["sphere", "cube", "cylinder", "torus", "cone"],
                        key=f"geometry_{idx}"
                    )
                    st.markdown("""</div>""", unsafe_allow_html=True)
                
                with control_cols[1]:
                    st.markdown("""
                    <div class="control-group">
                        <h4>Environment</h4>
                    """, unsafe_allow_html=True)
                    env_map = st.selectbox(
                        "",
                        ["studio", "outdoor", "night"],
                        key=f"env_{idx}"
                    )
                    st.markdown("""</div>""", unsafe_allow_html=True)
                
                with control_cols[2]:
                    st.markdown("""
                    <div class="control-group">
                        <h4>Lighting</h4>
                    """, unsafe_allow_html=True)
                    lighting = st.selectbox(
                        "",
                        ["studio", "outdoor", "dramatic", "product"],
                        key=f"lighting_{idx}"
                    )
                    st.markdown("""</div>""", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Material Channels
            if material["properties"].get("texture_maps"):
                st.markdown('<div class="channel-grid">', unsafe_allow_html=True)
                for map_type in material["properties"]["texture_maps"]:
                    st.markdown(f"""
                    <div class="channel-tile">
                        <img src="{material.get('preview_image', 'https://via.placeholder.com/150x100.png')}?text={map_type}" alt="{map_type} Map">
                        <h5>{map_type}</h5>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with preview_col2:
            # Material Properties
            st.markdown("""
            <div class="material-properties">
                <h4>Material Properties</h4>
            """, unsafe_allow_html=True)
            
            # Display all numeric properties
            for prop, value in material["properties"].items():
                if isinstance(value, (int, float)):
                    st.markdown(f"""
                    <div class="property-row">
                        <label>{prop.title()}</label>
                        <div class="value">{value:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Export Options
            st.markdown("""
            <div class="export-options">
                <h4>Export Options</h4>
                <button class="quick-action-btn">Export for Unity</button>
                <button class="quick-action-btn">Export for Unreal</button>
                <button class="quick-action-btn">Export for Blender</button>
            </div>
            """, unsafe_allow_html=True)
        
        # Update material on control changes
        st.markdown(f"""
        <script>
            if (window[`materialViewer{idx}`]) {{
                window[`materialViewer{idx}`].changeGeometry('{geometry}');
                window[`materialViewer{idx}`].loadEnvironmentMap('{env_map}');
                window[`materialViewer{idx}`].setLightingPreset('{lighting}');
            }}
        </script>
        """, unsafe_allow_html=True)

def submit_feedback(material_id: str, feedback_type: str):
    """Submit user feedback to backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/feedback",
            json={
                "material_id": material_id,
                "feedback_type": feedback_type,
                "reason": "User feedback from UI"
            },
            timeout=10
        )
        if response.status_code == 200:
            st.session_state.feedback_submitted = True
            st.success("Feedback submitted successfully!")
        else:
            st.error("Failed to submit feedback")
    except requests.exceptions.RequestException:
        st.error("Couldn't connect to feedback service")

if __name__ == "__main__":
    main()