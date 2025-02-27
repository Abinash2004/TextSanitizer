# Text Sanitizer Model

## Overview
The Text Sanitizer Model is a machine learning-based application designed to detect and censor explicit words in text while handling spelling variations. This model is trained on a dataset of explicit sentences and their fair replacements to ensure high accuracy in sanitization.

The application is built using Flask as the backend and integrates a pre-trained ML model. Users can input a paragraph containing explicit words, and the system will process and return a sanitized version.

## Features
- Detects and censors explicit words with variations.
- Trained on a dataset for high accuracy.
- Flask-based web interface for easy text sanitization.
- Supports cloud-based backend processing.
- Outputs sanitized text with an option to copy the results.

## Installation & Setup
### 1. Clone the Repository

git clone https://github.com/Abinash2004/TextSanitizer.git

cd text-sanitizer-model

### 2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt


### 4. Download and Setup Required Model Files
The model files required for this application are large and couldn't be uploaded to GitHub. You need to manually download them and place them in the correct directory.

#### Steps:
1. Download two ZIP folders from the provided Google Drive link:
   
   - https://drive.google.com/drive/folders/1yfJi6xpFDLbfUwQkGI-EFvun0UwXJEFy?usp=sharing

3. Extract both ZIP folders after downloading.
4. Move the extracted folders (sanitizer_model_250 & sanitizer_model_650) inside the "render" folder of the project.

   text-sanitizer-model/
   ├── render/
   │   ├── sanitizer_model_250/
   │   ├── sanitizer_model_650/
   
5. If these folders are missing, the model will not work.

### 5. Run the Flask Application

python app.py

The Flask app should now be running locally. Open your browser and go to:

http://127.0.0.1:5000/


## Usage
1. Enter a paragraph containing explicit words.
2. Click the Sanitize button.
3. The sanitized text will be displayed.
4. Use the Copy button to copy the sanitized output.

## Contributing
If you would like to contribute to this project, feel free to submit a pull request or report issues.
