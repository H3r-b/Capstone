# Malnutrition Detection System

A comprehensive AI-powered web application that leverages computer vision and machine learning to detect malnutrition from photographs. The system uses facial and body landmark analysis to assess nutritional status and provides severity classification for early intervention.

![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![React](https://img.shields.io/badge/React-19.1.1-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-teal.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Malnutrition Detection System is designed to assist healthcare professionals and researchers in identifying malnutrition through automated image analysis. By extracting facial and body landmarks using MediaPipe, the system analyzes physical characteristics that correlate with nutritional status and provides actionable insights.

### Key Capabilities

- **Image-based Detection**: Upload or capture images for instant analysis
- **Facial Feature Extraction**: Analyzes 468 facial landmarks for malnutrition indicators
- **Body Pose Analysis**: Evaluates body proportions and limb ratios
- **Severity Assessment**: Provides detailed severity classification when malnutrition is detected
- **Real-time Processing**: Fast API response times for quick assessments

##  Features

### Frontend Features

-  **Multiple Input Methods**
  - Drag and drop image upload
  - File browser selection
  - Webcam capture functionality

-  **Interactive UI**
  - Modern, responsive design with Tailwind CSS
  - Real-time result visualization
  - Color-coded severity indicators
  - Animated result cards

-  **Analysis Results**
  - Severity level classification
  - Confidence scores
  - Face detection status
  - Detailed recommendations

-  **Severity Assessment Form**
  - Automatic form display when malnutrition is detected
  - Input fields for age, height, weight, and sex
  - Real-time severity calculation

### Backend Features

-  **Machine Learning Pipeline**
  - XGBoost-based classification model
  - MediaPipe for landmark extraction
  - StandardScaler for feature normalization

-  **RESTful API**
  - FastAPI framework for high performance
  - CORS enabled for frontend integration
  - Comprehensive error handling
  - File upload support

-  **Model Evaluation**
  - Multiple performance metrics
  - Feature importance analysis
  - Correlation heatmaps
  - ROC curve visualization

##  Technology Stack

### Backend

- **Python 3.8+**
- **FastAPI** - Modern, fast web framework
- **MediaPipe** - Face mesh and pose detection
- **XGBoost** - Gradient boosting classifier
- **OpenCV** - Image processing
- **NumPy & Pandas** - Data manipulation
- **Scikit-learn** - Data preprocessing
- **Uvicorn** - ASGI server

### Frontend

- **React 19.1.1** - UI library
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Axios** - HTTP client
- **Chart.js** - Data visualization (if needed)

##  Architecture

```
┌─────────────────┐
│   React Frontend │
│   (Port 5173)    │
└────────┬─────────┘
         │ HTTP/REST
         │
┌────────▼─────────┐
│  FastAPI Backend │
│   (Port 8000)    │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────────┐
│MediaPipe│ │ XGBoost  │
│Landmarks│ │  Model   │
└────────┘ └──────────┘
```

##  Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm
- Webcam (optional, for camera capture)

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd Capstone/backend
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Ensure model files are present:**
   - `models/malnutrition_model.pkl`
   - `models/malnutrition_severity_model.pkl`

6. **Create uploads directory:**
   ```bash
   mkdir uploads
   ```

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd Capstone/frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

##  Usage

### Starting the Backend Server

1. **Navigate to backend directory:**
   ```bash
   cd Capstone/backend
   ```

2. **Activate virtual environment** (if not already activated)

3. **Run the FastAPI server:**
   ```bash
   uvicorn api:app --reload --port 8000
   ```

   The API will be available at `http://127.0.0.1:8000`

### Starting the Frontend Development Server

1. **Navigate to frontend directory:**
   ```bash
   cd Capstone/frontend
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:5173`

### Using the Application

1. **Upload or Capture Image:**
   - Click the upload area to browse for an image
   - Drag and drop an image file
   - Click "Take Photo" to capture from webcam

2. **Analyze Image:**
   - Click "Analyze Image" button
   - Wait for the analysis to complete

3. **View Results:**
   - Review severity level and confidence score
   - Check face detection status

4. **Severity Assessment (if malnutrition detected):**
   - Fill in the form with age, height, weight, and sex
   - Click "Calculate Severity" to get detailed classification

##  API Documentation

### Endpoints

#### `GET /`
Root endpoint to check API status.

**Response:**
```json
{
  "message": "Malnutrition Severity Prediction API is running."
}
```

#### `POST /predict`
Analyze an uploaded image for malnutrition detection.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Image file (required)
  - `age`: Age in years (optional, required for severity)
  - `height`: Height in cm (optional, required for severity)
  - `weight`: Weight in kg (optional, required for severity)
  - `sex`: Sex (1 for male, 0 for female) (optional, required for severity)

**Response:**
```json
{
  "filename": "image.jpg",
  "severity_score": 0.95,
  "severity_level": "Malnourished",
  "face_detected": true,
  "malnutrition_type": "Severe" // Only if malnutrition detected and severity fields provided
}
```

#### `POST /image`
Analyze an image from camera capture (simplified endpoint).

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Image file (required)

**Response:**
```json
{
  "filename": "capture.jpg",
  "severity_score": 1.0,
  "severity_level": "Healthy",
  "face_detected": true
}
```

#### `POST /severity`
Calculate malnutrition severity based on physical measurements.

**Request:**
- Method: `POST`
- Content-Type: `application/json`
- Body:
```json
{
  "age": 25,
  "height": 170.0,
  "weight": 55.0,
  "sex": 1
}
```

**Response:**
```json
{
  "severity": "Malnutrition" // or "No Malnutrition"
}
```

### API Documentation (Swagger)

Once the backend server is running, visit:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

##  Project Structure

```
Capstone/
├── backend/
│   ├── api.py                 # FastAPI application and endpoints
│   ├── main.py                # Standalone prediction script
│   ├── malnutrition_predictor.py  # Core ML model class
│   ├── requirements.txt       # Python dependencies
│   ├── models/                # Trained model files
│   │   ├── malnutrition_model.pkl
│   │   └── malnutrition_severity_model.pkl
│   ├── utils/                 # Utility functions
│   │   ├── inference.py
│   │   └── model.json
│   ├── tests/                 # Test files
│   │   └── test_api.py
│   └── uploads/               # Temporary file storage
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main React component
│   │   ├── App.css            # Component styles
│   │   ├── main.jsx           # React entry point
│   │   └── index.css          # Global styles
│   ├── public/                # Static assets
│   ├── package.json           # Node.js dependencies
│   ├── vite.config.js         # Vite configuration
│   ├── tailwind.config.js     # Tailwind CSS configuration
│   └── index.html             # HTML template
│
└── README.md                  # This file
```

##  Model Details

### Feature Extraction

The system extracts **34 features** from images:

#### Facial Features (13 features)
- Face width and length
- Face ratio
- Cheek depth (left and right)
- Eye socket depth (left and right)
- Jaw width (top and bottom)
- Temple measurements (left and right)
- Statistical features (mean, std)

#### Body Features (21 features)
- Shoulder width, torso length, hip width
- Arm measurements (upper and lower, left and right)
- Leg measurements (upper and lower, left and right)
- Body ratios (shoulder-hip, torso-shoulder)
- Limb ratios (arm and leg)
- Visibility scores
- Statistical features

### Machine Learning Models

1. **Malnutrition Detection Model**
   - Algorithm: XGBoost Classifier
   - Task: Binary classification (Healthy vs. Malnourished)
   - Input: 34 extracted features
   - Output: Classification and confidence score

2. **Severity Assessment Model**
   - Algorithm: XGBoost Classifier
   - Task: Multi-class classification
   - Input: Age, height, weight, sex
   - Output: Severity level classification

### Model Performance

The models are trained on curated datasets and evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

##  Development

### Running Tests

```bash
cd Capstone/backend
pytest tests/
```

### Building for Production

**Frontend:**
```bash
cd Capstone/frontend
npm run build
```

**Backend:**
The FastAPI backend can be deployed using:
- Uvicorn (production server)
- Gunicorn with Uvicorn workers
- Docker containers
- Cloud platforms (AWS, Azure, GCP)

##  Troubleshooting

### Common Issues

1. **Model files not found:**
   - Ensure model files are in `backend/models/` directory
   - Check file paths in `api.py`

2. **CORS errors:**
   - Verify CORS middleware is configured in `api.py`
   - Check frontend API endpoint URLs

3. **Face not detected:**
   - Ensure image contains clear face and body
   - Try different image angles or lighting
   - Check image quality and resolution

4. **Camera not working:**
   - Grant camera permissions in browser
   - Use HTTPS for camera access (required by browsers)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Note:** By contributing to this project, you agree that your contributions will be licensed under the GNU General Public License v3.0 (GPLv3). This ensures that all derivative works remain open source and freely available.

## ⚠️ Disclaimer

This system is intended for research and educational purposes. It should not be used as a sole diagnostic tool. Always consult with healthcare professionals for medical diagnosis and treatment decisions.

## 📄 License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html) for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

##  Authors

- Herbert George
- Kamal Adithya 
- Vineeta Pareek
- Gowtham B

##  Acknowledgments

- MediaPipe team for the landmark detection framework
- XGBoost developers for the machine learning library
- FastAPI team for the excellent web framework
- React community for the robust UI library

##  Contact

For questions or support, please open an issue on the repository.

---

**Made with ❤️ for better healthcare outcomes**
