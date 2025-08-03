// Global variables
let uploadedFile = null;
let analysisResults = null;

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const imageFilename = document.getElementById('imageFilename');
const imageSize = document.getElementById('imageSize');
const imageDimensions = document.getElementById('imageDimensions');
const analysisSection = document.getElementById('analysisSection');
const loadingSpinner = document.getElementById('loadingSpinner');
const results = document.getElementById('results');

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

function initializeEventListeners() {
    // File input change event
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Prevent default drag behaviors on document
    document.addEventListener('dragover', function(e) {
        e.preventDefault();
    });
    
    document.addEventListener('drop', function(e) {
        e.preventDefault();
    });
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect({ target: { files: files } });
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    
    if (!file) {
        return;
    }
    
    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
        showAlert('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP, WEBP)', 'error');
        return;
    }
    
    // Validate file size (16MB max)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showAlert('File size exceeds 16MB limit', 'error');
        return;
    }
    
    uploadedFile = file;
    uploadFile(file);
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Show progress
    uploadProgress.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Uploading...';
    
    // Simulate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 20;
        if (progress > 90) progress = 90;
        progressFill.style.width = progress + '%';
    }, 100);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        clearInterval(progressInterval);
        
        if (data.success) {
            progressFill.style.width = '100%';
            progressText.textContent = 'Upload complete!';
            
            setTimeout(() => {
                uploadProgress.style.display = 'none';
                showImagePreview(file, data.filename);
            }, 1000);
        } else {
            showAlert(data.error || 'Upload failed', 'error');
            uploadProgress.style.display = 'none';
        }
    })
    .catch(error => {
        clearInterval(progressInterval);
        console.error('Upload error:', error);
        showAlert('Upload failed. Please try again.', 'error');
        uploadProgress.style.display = 'none';
    });
}

function showImagePreview(file, filename) {
    // Create file reader to display image
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        
        // Create image to get dimensions
        const img = new Image();
        img.onload = function() {
            imageDimensions.textContent = `${this.width} × ${this.height} pixels`;
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    // Set file info
    imageFilename.textContent = file.name;
    imageSize.textContent = formatFileSize(file.size);
    
    // Show preview section
    imagePreview.style.display = 'block';
    imagePreview.scrollIntoView({ behavior: 'smooth' });
    
    // Store filename for analysis
    uploadedFile.serverFilename = filename;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function analyzeImage() {
    if (!uploadedFile || !uploadedFile.serverFilename) {
        showAlert('No file uploaded for analysis', 'error');
        return;
    }
    
    // Show analysis section
    analysisSection.style.display = 'block';
    loadingSpinner.style.display = 'block';
    results.style.display = 'none';
    
    // Scroll to analysis section
    analysisSection.scrollIntoView({ behavior: 'smooth' });
    
    // Animate analysis steps
    animateAnalysisSteps();
    
    // Send analysis request
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            filename: uploadedFile.serverFilename
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showAlert(data.error, 'error');
            analysisSection.style.display = 'none';
        } else {
            analysisResults = data;
            displayResults(data);
        }
    })
    .catch(error => {
        console.error('Analysis error:', error);
        showAlert('Analysis failed. Please try again.', 'error');
        analysisSection.style.display = 'none';
    });
}

function animateAnalysisSteps() {
    const steps = document.querySelectorAll('.step');
    steps.forEach(step => step.classList.remove('active'));
    
    let currentStep = 0;
    const stepInterval = setInterval(() => {
        if (currentStep < steps.length) {
            steps[currentStep].classList.add('active');
            currentStep++;
        } else {
            clearInterval(stepInterval);
        }
    }, 2000);
}

function displayResults(data) {
    // Hide loading, show results
    loadingSpinner.style.display = 'none';
    results.style.display = 'block';
    
    // Display description
    const descriptionContent = document.getElementById('descriptionContent');
    descriptionContent.textContent = data.description;
    
    // Display materials
    displayMaterials(data.analysis);
    
    // Display metrics
    displayMetrics(data.analysis);
    
    // Display StableMaterials compatibility if available
    displayStableMaterialsInfo(data.analysis);
    
    // Scroll to results
    results.scrollIntoView({ behavior: 'smooth' });
}

function displayMaterials(analysis) {
    const materialsGrid = document.getElementById('materialsGrid');
    materialsGrid.innerHTML = '';
    
    if (analysis.error) {
        materialsGrid.innerHTML = `
            <div class="alert alert-error">
                <strong>Analysis Error:</strong> ${analysis.error}
                ${analysis.raw_response ? `<br><br><strong>Raw Response:</strong><br>${analysis.raw_response}` : ''}
            </div>
        `;
        return;
    }
    
    if (!analysis.materials || analysis.materials.length === 0) {
        materialsGrid.innerHTML = `
            <div class="alert alert-info">
                No materials were identified in the image.
            </div>
        `;
        return;
    }
    
    analysis.materials.forEach(material => {
        const materialCard = createMaterialCard(material);
        materialsGrid.appendChild(materialCard);
    });
}

function createMaterialCard(material) {
    const card = document.createElement('div');
    card.className = 'material-card';
    
    const confidence = Math.round((material.confidence || 0) * 100);
    const properties = material.properties || {};
    
    card.innerHTML = `
        <div class="material-header">
            <div class="material-name">${material.name || 'Unknown Material'}</div>
            <div class="confidence-badge">${confidence}%</div>
        </div>
        
        <div class="material-category">${material.category || 'Unknown Category'}</div>
        
        <div class="properties-grid">
            <div class="property roughness-bar">
                <div class="property-label">Roughness</div>
                <div class="property-value">${properties.roughness || 'N/A'}</div>
                ${properties.roughness ? `<div class="property-bar"><div class="property-bar-fill" style="width: ${(properties.roughness * 100)}%"></div></div>` : ''}
            </div>
            
            <div class="property metallic-bar">
                <div class="property-label">Metallic</div>
                <div class="property-value">${properties.metallic || 'N/A'}</div>
                ${properties.metallic ? `<div class="property-bar"><div class="property-bar-fill" style="width: ${(properties.metallic * 100)}%"></div></div>` : ''}
            </div>
            
            <div class="property">
                <div class="property-label">Reflectance</div>
                <div class="property-value">${properties.reflectance || 'N/A'}</div>
            </div>
            
            <div class="property">
                <div class="property-label">Base Color</div>
                <div class="property-value">
                    ${properties.basecolor ? `<span style="display: inline-block; width: 20px; height: 20px; background: ${properties.basecolor}; border-radius: 3px; margin-right: 8px; vertical-align: middle;"></span>${properties.basecolor}` : 'N/A'}
                </div>
            </div>
        </div>
        
        <div class="material-description">
            ${material.description || 'No description available.'}
        </div>
        
        ${material.applications && material.applications.length > 0 ? `
            <div class="applications">
                ${material.applications.map(app => `<span class="application-tag">${app}</span>`).join('')}
            </div>
        ` : ''}
        
        ${material.pbr_characteristics ? `
            <div style="margin-top: 15px; padding: 15px; background: rgba(40, 167, 69, 0.1); border-radius: 8px;">
                <strong>PBR Characteristics:</strong>
                <div style="margin-top: 10px;">
                    <div><strong>Normal Map:</strong> ${material.pbr_characteristics.normal_map || 'N/A'}</div>
                    <div><strong>Height Variation:</strong> ${material.pbr_characteristics.height_variation || 'N/A'}</div>
                    <div><strong>Specular Behavior:</strong> ${material.pbr_characteristics.specular_behavior || 'N/A'}</div>
                </div>
            </div>
        ` : ''}
        
        ${material.matsynth_reference ? `
            <div style="margin-top: 15px; padding: 15px; background: rgba(118, 75, 162, 0.1); border-radius: 8px;">
                <strong>MatSynth Reference:</strong>
                <div style="margin-top: 10px; font-size: 0.9rem;">${material.matsynth_reference}</div>
            </div>
        ` : ''}
        
        ${material.database_info && Object.keys(material.database_info).length > 0 ? `
            <div style="margin-top: 15px; padding: 15px; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                <strong>Enhanced Database Info:</strong>
                ${material.database_info.stable_materials_enhanced ? `
                    <div style="margin-top: 10px;">
                        <strong>StableMaterials Enhanced:</strong>
                        <div style="font-size: 0.9rem; margin-top: 5px;">
                            Source: ${material.database_info.stable_materials_enhanced.source || 'N/A'}
                        </div>
                    </div>
                ` : ''}
                <details style="margin-top: 10px;">
                    <summary style="cursor: pointer; font-weight: bold;">View Full Database Info</summary>
                    <pre style="margin-top: 10px; font-size: 0.8rem; overflow-x: auto;">${JSON.stringify(material.database_info, null, 2)}</pre>
                </details>
            </div>
        ` : ''}
    `;
    
    return card;
}

function displayMetrics(analysis) {
    const metricsGrid = document.getElementById('metricsGrid');
    metricsGrid.innerHTML = '';
    
    const metrics = analysis.quality_metrics || {};
    
    // Create enhanced metric cards
    const metricCards = [
        {
            label: 'Materials Found',
            value: analysis.materials ? analysis.materials.length : 0,
            icon: 'fas fa-cubes'
        },
        {
            label: 'Avg Confidence',
            value: analysis.materials ? Math.round(analysis.materials.reduce((sum, m) => sum + (m.confidence || 0), 0) / analysis.materials.length * 100) + '%' : '0%',
            icon: 'fas fa-chart-line'
        },
        {
            label: 'Image Quality',
            value: metrics.resolution_estimate || 'Unknown',
            icon: 'fas fa-image'
        },
        {
            label: 'Clarity Score',
            value: metrics.clarity ? Math.round(metrics.clarity * 100) + '%' : 'N/A',
            icon: 'fas fa-eye'
        },
        {
            label: 'Lighting Quality',
            value: metrics.lighting_quality ? Math.round(metrics.lighting_quality * 100) + '%' : 'N/A',
            icon: 'fas fa-lightbulb'
        },
        {
            label: 'Material Visibility',
            value: metrics.material_visibility ? Math.round(metrics.material_visibility * 100) + '%' : 'N/A',
            icon: 'fas fa-search'
        }
    ];
    
    metricCards.forEach(metric => {
        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML = `
            <div class="metric-value">${metric.value}</div>
            <div class="metric-label">
                <i class="${metric.icon}"></i> ${metric.label}
            </div>
        `;
        metricsGrid.appendChild(card);
    });
}

function displayStableMaterialsInfo(analysis) {
    // Check if we already have a StableMaterials section
    let stableMaterialsSection = document.getElementById('stableMaterialsSection');
    
    if (!stableMaterialsSection) {
        // Create the section if it doesn't exist
        stableMaterialsSection = document.createElement('div');
        stableMaterialsSection.id = 'stableMaterialsSection';
        stableMaterialsSection.className = 'stable-materials-section';
        stableMaterialsSection.innerHTML = `
            <h3><i class="fas fa-magic"></i> StableMaterials Analysis</h3>
            <div id="stableMaterialsContent"></div>
        `;
        
        // Insert after the metrics section
        const metricsSection = document.querySelector('.metrics-section');
        if (metricsSection) {
            metricsSection.parentNode.insertBefore(stableMaterialsSection, metricsSection.nextSibling);
        }
    }
    
    const stableMaterialsContent = document.getElementById('stableMaterialsContent');
    
    if (analysis.stable_materials_compatibility) {
        stableMaterialsContent.innerHTML = `
            <div class="stable-materials-compatibility">
                <div style="padding: 20px; background: rgba(118, 75, 162, 0.1); border-radius: 10px; border-left: 4px solid #764ba2;">
                    <h4><i class="fas fa-cogs"></i> Material Generation Compatibility</h4>
                    <p style="margin-top: 10px; line-height: 1.6;">${analysis.stable_materials_compatibility}</p>
                </div>
            </div>
        `;
    } else {
        stableMaterialsContent.innerHTML = `
            <div class="stable-materials-info">
                <div style="padding: 20px; background: rgba(118, 75, 162, 0.05); border-radius: 10px;">
                    <h4><i class="fas fa-info-circle"></i> StableMaterials Integration</h4>
                    <p style="margin-top: 10px; line-height: 1.6;">
                        This analysis is enhanced with StableMaterials knowledge base, providing accurate PBR 
                        property estimations based on 4,000+ high-resolution material samples from the MatSynth dataset.
                    </p>
                    <div style="margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap;">
                        <span class="tech-badge" style="background: rgba(118, 75, 162, 0.2); color: #764ba2; padding: 5px 12px; border-radius: 15px; font-size: 0.9rem;">
                            <i class="fas fa-database"></i> MatSynth Dataset
                        </span>
                        <span class="tech-badge" style="background: rgba(118, 75, 162, 0.2); color: #764ba2; padding: 5px 12px; border-radius: 15px; font-size: 0.9rem;">
                            <i class="fas fa-magic"></i> StableMaterials AI
                        </span>
                        <span class="tech-badge" style="background: rgba(118, 75, 162, 0.2); color: #764ba2; padding: 5px 12px; border-radius: 15px; font-size: 0.9rem;">
                            <i class="fas fa-cube"></i> PBR Analysis
                        </span>
                    </div>
                </div>
            </div>
        `;
    }
}

function exportResults() {
    if (!analysisResults) {
        showAlert('No results to export', 'error');
        return;
    }
    
    const exportData = {
        timestamp: new Date().toISOString(),
        filename: analysisResults.filename,
        description: analysisResults.description,
        analysis: analysisResults.analysis
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `material_analysis_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    showAlert('Results exported successfully!', 'success');
}

function showAlert(message, type) {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create new alert
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.innerHTML = `
        <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : type === 'success' ? 'check-circle' : 'info-circle'}"></i>
        ${message}
    `;
    
    // Insert at top of container
    const container = document.querySelector('.container');
    container.insertBefore(alert, container.firstChild);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        alert.remove();
    }, 5000);
    
    // Scroll to alert
    alert.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Utility function to format JSON for display
function formatJSON(obj) {
    return JSON.stringify(obj, null, 2);
}

// Add smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
}); 