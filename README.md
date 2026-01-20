# Brazilian Vehicle License Plate Recognition

End-to-end computer vision system for detecting and recognizing Brazilian vehicle license plates (Mercosul and legacy formats) using OpenCV and OCR techniques.

## Overview
This project implements a complete pipeline for vehicle license plate recognition, combining classical computer vision methods, OCR engines and post-processing heuristics to achieve robust performance in real-world scenarios.

The system was designed to handle challenging conditions such as low contrast, noise, perspective distortion and different Brazilian plate standards.

## Technologies
- Python
- OpenCV
- NumPy
- Tesseract OCR
- EasyOCR
- YOLOv8 (optional detection support)

## Features
- Detection of Brazilian license plates (Mercosul and old standard)
- Multiple detection strategies:
  - Contour-based detection
  - Connected components analysis
  - Edge-based detection
- Advanced image preprocessing:
  - CLAHE (contrast enhancement)
  - Adaptive and Otsu thresholding
  - Morphological operations
- OCR ensemble (Tesseract + EasyOCR)
- Intelligent post-processing and character correction
- Confidence score for detected plates
- Simple execution script for desktop usage

## Pipeline
1. Image preprocessing and region of interest selection  
2. Candidate plate detection using multiple strategies  
3. Geometric and heuristic filtering  
4. OCR validation and character isolation  
5. OCR ensemble and post-processing  
6. Final validation and confidence scoring  

## How to Run
Install dependencies:
```bash
pip install -r requirements.txt
