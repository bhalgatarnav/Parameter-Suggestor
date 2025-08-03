# Material Property Analyzer Platform

A comprehensive web platform that analyzes uploaded images to identify materials and their properties using advanced AI models including joy-caption-alpha-two, OpenAI GPT-4, and insights from MatSynth dataset.

## 🚀 Features

- **Image Upload & Analysis**: Upload images via drag-and-drop or file browser
- **AI-Powered Description**: Generate detailed image descriptions using HuggingFace's joy-caption-alpha-two model
- **Material Identification**: Identify individual materials in images using GPT-4 analysis
- **PBR Property Analysis**: Estimate Physically Based Rendering properties including:
  - Roughness values (0-1 scale)
  - Metallic properties (0-1 scale)
  - Base color/Albedo
  - Reflectance values
  - Normal map characteristics
- **Quality Metrics**: Analyze image quality, clarity, and lighting conditions
- **Material Database**: Enhanced analysis with material property databases
- **Export Results**: Download analysis results as JSON files
- **Beautiful UI**: Modern, responsive interface with real-time progress indicators

## 🛠 Technology Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **AI Models**: 
  - OpenAI GPT-4 for material analysis
  - HuggingFace joy-caption-alpha-two for image description
  - MatSynth dataset for material properties
- **UI Framework**: Custom CSS with Font Awesome icons
- **File Handling**: Werkzeug secure file uploads

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (provided in the setup)
- Internet connection for AI model access

## 🔧 Installation & Setup

1. **Clone the repository** (if applicable) or ensure you have all the files in your project directory:
   ```bash
   cd Parameter-Suggestor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary directories** (they should already exist):
   ```bash
   mkdir -p static/uploads static/css static/js templates
   ```

4. **Verify file structure**:
   ```
   Parameter-Suggestor/
   ├── app.py
   ├── requirements.txt
   ├── README.md
   ├── templates/
   │   └── index.html
   └── static/
       ├── css/
       │   └── style.css
       ├── js/
       │   └── script.js
       └── uploads/ (created automatically)
   ```

5. **API Configuration**: The OpenAI API key is already configured in `app.py`. If you need to change it, edit the `openai.api_key` variable in `app.py`.

## 🚀 Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the application**:
   Open your web browser and go to: `http://localhost:8080`

3. **Upload and analyze images**:
   - Drag and drop an image or click "Browse Files"
   - Supported formats: PNG, JPG, JPEG, GIF, BMP, WEBP
   - Maximum file size: 16MB
   - Click "Analyze Materials" to start the analysis

## 📊 How It Works

### Analysis Pipeline

1. **Image Upload**: Secure file upload with validation
2. **Image Description**: joy-caption-alpha-two generates detailed image descriptions
3. **Material Analysis**: GPT-4 analyzes the description to identify materials and properties
4. **Database Enhancement**: Material properties are enhanced with database information
5. **Results Display**: Comprehensive analysis results with visual metrics

### Material Properties Analyzed

- **Physical Properties**:
  - Surface roughness
  - Metallic/non-metallic classification
  - Reflectance characteristics
  - Base color and albedo values

- **PBR Properties**:
  - Roughness maps
  - Metallic maps
  - Normal map characteristics
  - Specular reflectance

- **Quality Metrics**:
  - Image resolution estimate
  - Clarity scores
  - Lighting quality
  - Confidence ratings

## 🎯 Use Cases

- **Computer Graphics**: Generate PBR materials for 3D rendering
- **Game Development**: Analyze reference materials for game assets
- **Architectural Visualization**: Identify building materials and properties
- **Material Science**: Research and analysis of material properties
- **Digital Content Creation**: Enhance textures and materials

## 📁 API Integration Details

### Models Used

1. **joy-caption-alpha-two** (HuggingFace):
   - Generates detailed image descriptions
   - Includes lighting, composition, and quality analysis
   - API endpoint: `fancyfeast/joy-caption-alpha-two`

2. **OpenAI GPT-4**:
   - Analyzes image descriptions for materials
   - Estimates PBR properties
   - Provides confidence scores and applications

3. **MatSynth Dataset** (Reference):
   - 4,000+ ultra-high resolution PBR materials
   - Physically based rendering properties
   - Material categorization and metadata

## 🔧 Configuration Options

### File Upload Settings
- Modify `MAX_FILE_SIZE` in `app.py` to change upload limits
- Edit `ALLOWED_EXTENSIONS` to support additional file types

### AI Model Configuration
- Update OpenAI API key in `app.py`
- Modify GPT-4 parameters (temperature, max_tokens) in `analyze_materials_with_gpt()`
- Customize joy-caption options in `get_image_description()`

## 🎨 UI Customization

The platform uses a modern gradient design with:
- Responsive layout for all device sizes
- Drag-and-drop file upload interface
- Real-time progress indicators
- Animated analysis steps
- Interactive material property cards

### Color Scheme
- Primary gradient: `#667eea` to `#764ba2`
- Success colors: `#28a745` to `#20c997`
- Error handling with contextual alerts

## 📊 Export Features

Results can be exported as JSON files containing:
- Original filename and timestamp
- Complete image description
- Detailed material analysis
- Property estimations
- Confidence scores
- Quality metrics

## 🛡 Security Features

- Secure filename handling with Werkzeug
- File type validation
- File size limits
- Input sanitization
- CSRF protection (Flask secret key)

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Errors**:
   - Verify API key is correct
   - Check API quota and billing
   - Ensure internet connectivity

2. **HuggingFace Model Issues**:
   - Model may be temporarily unavailable
   - Check internet connection
   - Try again after a few minutes

3. **File Upload Issues**:
   - Ensure file size is under 16MB
   - Check file format is supported
   - Verify `static/uploads` directory exists

4. **Python Dependencies**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## 📈 Performance Notes

- **Analysis Time**: 30-60 seconds depending on image complexity
- **Concurrent Users**: Single-threaded Flask development server
- **Production**: Use WSGI server (Gunicorn, uWSGI) for production deployment

## 🔮 Future Enhancements

- Integration with StableMaterials model for direct PBR generation
- Real-time material property visualization
- Batch processing for multiple images
- Advanced filtering and search capabilities
- Database persistence for analysis history
- API endpoints for programmatic access

## 📄 License

This project integrates with various open-source and commercial services:
- MatSynth dataset (CC0 and CC-BY licenses)
- OpenAI GPT-4 (Commercial API)
- HuggingFace models (Various licenses)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional AI model integrations
- Enhanced material databases
- Performance optimizations
- UI/UX improvements
- Mobile responsiveness enhancements

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure API keys are configured correctly
4. Check network connectivity for AI model access

---

Built with ❤️ using Flask, OpenAI, HuggingFace, and modern web technologies. 