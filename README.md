# Inteligência Artificial em Microscopia - Flask Application

An educational Flask web application designed to teach artificial intelligence concepts in microscopy through interactive phases. Students learn how AI can enhance images, learn from examples, and detect objects through three progressive phases.

## Table of Contents

1. [Installation Instructions](#installation-instructions)
2. [Running the Application](#running-the-application)
3. [How to Interact with the Application](#how-to-interact-with-the-application)
4. [Application Structure](#application-structure)
5. [Troubleshooting](#troubleshooting)

## Installation Instructions

### Prerequisites

Before installing the application, ensure you have the following installed on your system:

- **Python 3.10 or higher** - [Download Python](https://www.python.org/downloads/) OR **Anaconda/Miniconda** - [Download Anaconda](https://www.anaconda.com/products/distribution) or [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **pip** (Python package installer) - Usually comes with Python
- **Git** (optional, for cloning the repository) - [Download Git](https://git-scm.com/downloads)

### Step 1: Clone or Download the Repository

**Option A: Using Git (Recommended)**
```bash
git clone https://github.com/brunomsaraiva/NoiteDosInvestigadores25.git
cd NoiteDosInvestigadores25
```

**Option B: Download ZIP**
1. Download the project as a ZIP file from GitHub
2. Extract the ZIP file to your desired location
3. Navigate to the extracted folder using terminal/command prompt

### Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment helps isolate the project dependencies from your system Python installation.

#### Option A: Using Python venv (Standard Library)

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt, indicating the virtual environment is active.

#### Option B: Using Conda (Anaconda/Miniconda)

If you have Anaconda or Miniconda installed, you can use conda to create and manage your environment:

**Create a new conda environment:**
```bash
conda create -n microscopy-ai python=3.10
```

**Activate the conda environment:**
```bash
conda activate microscopy-ai
```

**Alternative: Create environment with some packages pre-installed:**
```bash
conda create -n microscopy-ai python=3.10 numpy pillow
conda activate microscopy-ai
```

You should see `(microscopy-ai)` appear in your terminal prompt, indicating the conda environment is active.

**To deactivate the environment later:**
```bash
conda deactivate
```

### Step 3: Install Dependencies

Install all required Python packages using pip:

```bash
pip install -r requirements.txt
```

**Note for Conda users:** If you're using a conda environment, you can still use pip to install from requirements.txt, or alternatively install packages using conda where available:

```bash
# Option 1: Use pip (recommended for this project)
pip install -r requirements.txt

# Option 2: Mix conda and pip (install available packages with conda first)
conda install flask numpy pillow
pip install -r requirements.txt  # This will install remaining packages
```

This will install:
- Flask 2.3.3 (Web framework)
- Werkzeug 2.3.7 (WSGI utilities)
- Jinja2 3.1.2 (Template engine)
- opencv-python 4.8.1.78 (Computer vision library)
- Pillow 10.0.1 (Image processing)
- numpy 1.24.3 (Numerical computing)

### Step 4: Verify Installation

Ensure all dependencies are installed correctly:

```bash
pip list
```

You should see all the packages from requirements.txt listed.

## Running the Application

### Development Mode

1. **Ensure you're in the project directory and virtual environment is active:**
   
   **For Python venv:**
   ```bash
   cd NoiteDosInvestigadores25
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```
   
   **For Conda:**
   ```bash
   cd NoiteDosInvestigadores25
   conda activate microscopy-ai
   ```

2. **Start the Flask development server:**
   ```bash
   python app.py
   ```

3. **Access the application:**
   Open your web browser and navigate to:
   ```
   http://localhost:5001
   ```
   or
   ```
   http://127.0.0.1:5001
   ```

4. **Stop the server:**
   Press `Ctrl + C` in the terminal to stop the server.

### Production Mode

For production deployment, consider using a proper WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

## How to Interact with the Application

The application is designed as an educational journey through three phases of AI learning in microscopy. Here's how to use each section:

### Homepage

- **Starting Point:** The main page introduces students to AI in microscopy
- **Navigation:** Use the phase buttons to begin the learning journey
- **About Section:** Contains educational background information about AI and microscopy

### Phase 1: Image Enhancement with AI

**Objective:** Learn how AI can automatically improve microscopy images

**How to Interact:**
1. **View the Distorted Image:** See a microscopy image with common problems (blur, low contrast, noise)
2. **Use Manual Controls:** Try adjusting sliders for:
   - Sharpness (0-100)
   - Contrast (0-100) 
   - Brightness (0-100)
   - Noise Filter (0-100)
   - Zoom (50-150%)
3. **AI Enhancement:** Click "Melhorar com IA" to see automatic AI enhancement
4. **Compare Results:** The AI automatically finds optimal settings
5. **Completion:** A modal shows your manual score vs AI performance

**Educational Goal:** Understand that AI can automatically optimize image parameters that humans adjust manually.

### Phase 2: Training the AI with Examples

**Objective:** Learn how AI learns from labeled examples (supervised learning)

**How to Interact:**
1. **View Cell Images:** See microscopy images with cells to identify
2. **Manual Annotation:** 
   - Click on cells to mark them (green circles)
   - Click again to unmark cells
   - Use "Limpar Anotações" to reset
3. **Track Progress:**
   - **Accuracy:** How precisely you identify actual cells
   - **Progress:** How many cells you've attempted to find
4. **Submit Training:** Click "Terminar Anotação" when finished
5. **Results:** View your training performance in three categories:
   - Cells found
   - Accuracy percentage
   - Progress percentage

**Educational Goal:** Understand that AI quality depends on the quality and completeness of training data.

### Phase 3: AI Cell Detection

**Objective:** See how well the trained AI performs on new images

**How to Interact:**
1. **Image Processing:** Click "Melhorar com IA" to enhance the image
2. **AI Detection:** Click "IA Encontrar Células" to run detection
3. **View Results:**
   - See detected cells marked with green circles
   - View detection statistics
   - Training Quality score (Accuracy × Progress from Phase 2)
4. **Understand Performance:**
   - Number of cells detected depends on Phase 2 training quality
   - Better training = better AI performance
5. **Reset:** Use "Voltar ao Início" to try different scenarios

**Educational Goal:** Demonstrate how training quality directly impacts AI performance in real applications.

### Navigation Features

- **Home Button:** Return to the main page from any phase
- **Phase Navigation:** Jump between phases (though sequential completion is recommended)
- **Reset Buttons:** Start over within each phase
- **Progress Tracking:** Session maintains scores across phases

### Educational Flow

**Recommended Learning Sequence:**
1. **Phase 1 → Phase 2 → Phase 3:** Understand the complete AI pipeline
2. **Experimentation:** Try different approaches in Phase 2 and see effects in Phase 3
3. **Discussion Points:** 
   - Why did the AI perform better/worse?
   - How does training data quality affect results?
   - What are the limitations of AI in microscopy?

## Application Structure

```
NoiteDosInvestigadores25/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── static/
│   └── images/          # Microscopy images and assets
│       ├── cell_fluor_1.png
│       ├── cell_labels_1.png
│       └── ...
└── templates/           # HTML templates
    ├── base.html        # Base template
    ├── index.html       # Homepage
    ├── fase1.html       # Phase 1: Image Enhancement
    ├── fase2.html       # Phase 2: AI Training
    ├── fase3.html       # Phase 3: AI Detection
    └── about.html       # About page
```

## Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
Error: [Errno 48] Address already in use
```
**Solution:** Kill the process using port 5001 or use a different port:
```bash
lsof -ti:5001 | xargs kill -9
# or
python app.py --port 5002
```

**2. Module Not Found Errors**
```bash
ModuleNotFoundError: No module named 'flask'
```
**Solutions:**
- Ensure virtual environment is activated (see step 1 in Running the Application)
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version`
- **For Conda users:** Ensure conda environment is activated: `conda activate microscopy-ai`

**3. OpenCV Issues (macOS)**
```bash
ImportError: No module named 'cv2'
```
**Solution:**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

**4. Permission Errors**
```bash
PermissionError: [Errno 13] Permission denied
```
**Solution:**
- Use virtual environment
- On Unix systems: `chmod +x app.py`

**5. Image Loading Issues**
- Ensure `static/images/` directory exists
- Check image file permissions
- Verify image file formats are supported

### Getting Help

If you encounter issues:

1. **Check Error Messages:** Read the full error output in the terminal
2. **Verify Installation:** Ensure all requirements are installed correctly
3. **Check File Paths:** Ensure you're in the correct directory
4. **Browser Console:** Check for JavaScript errors in browser developer tools
5. **Restart:** Try restarting the Flask server

### Development Tips

- **Debug Mode:** The application runs in debug mode by default, showing detailed error pages
- **Auto-reload:** Changes to Python files automatically restart the server
- **Static Files:** Images and CSS are served from the `static/` directory
- **Session Data:** User progress is stored in browser sessions (clears on browser restart)

### System Requirements

- **Memory:** At least 512MB RAM for image processing
- **Storage:** ~50MB for application and dependencies
- **Browser:** Modern web browser (Chrome, Firefox, Safari, Edge)
- **Network:** No internet connection required after installation
