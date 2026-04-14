# Language Identification CNN and Spectogram

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![React](https://img.shields.io/badge/React-18.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)

An end-to-end Machine Learning pipeline designed for real-time Automatic Language Identification (LID). This system utilizes Spectral Noise Gating, Mel-Spectrogram feature extraction, and a 2D Convolutional Neural Network (CNN) to visually classify the phonetic signatures of human speech across five languages (English, French, German, Hindi, Spanish).

## Key Features

* **Real-Time Audio Processing:** Captures live microphone input via the browser and processes it through a sophisticated mathematical pipeline.
* **Spectral Noise Reduction:** Implements `noisereduce` to mathematically scrub continuous background interference (e.g., newsroom hum, static) before classification.
* **Computer Vision for Audio:** Converts 1D audio arrays into 2D Mel-Spectrogram images, allowing the CNN to classify the literal "shapes" of language formants.
* **Interactive Dashboard:** A professional React frontend featuring a live pipeline tracker, interactive audio playback, and real-time visualization of the generated spectrograms and prediction matrices.

## Tech Stack

**Frontend:**
* React.js (Vite)
* Recharts (Data Visualization)
* CSS3 (Glassmorphism UI)

**Backend & ML:**
* Python 3.x
* FastAPI & Uvicorn (REST API)
* TensorFlow / Keras (CNN Architecture)
* Librosa (Digital Signal Processing)
* Scikit-Learn & Pandas (Data Formatting)

## Project Structure

```text
ASP_PROJECT/
├── frontend-react/           # React dashboard UI and components
├── test_samples/             # Sample .mp3 and .wav files for testing
├── app.py                    # Main FastAPI backend server
├── download_real_data.py     # Script to fetch the FLEURS dataset
├── extract_cnn.py            # Extracts Mel-Spectrograms from raw audio
├── train_cnn.py              # Builds and trains the CNN model
├── Project_Report.md         # Detailed academic documentation
├── Report.docx               # Formatted project report
└── .gitignore                # Git exclusion rules

## Installation & Setup

1. Clone the Repository & Setup Backend

    1.1 Open your terminal and clone the repository:

        git clone https://github.com/shivamchoudhary62/Language_Identification_CNN
        cd ASP_PROJECT

    1.2 Create and activate a virtual environment (named asp):

        # On Windows
        python -m venv asp
        asp\Scripts\activate

        # On Mac/Linux
        python3 -m venv asp
        source asp/bin/activate

    1.3 Install the required Python dependencies:

        pip install fastapi uvicorn librosa noisereduce tensorflow pandas scikit-learn matplotlib


2. Generate the Deep Learning Models
Run the following scripts in order to download the data, extract the spectrograms, and train the AI brain:

    # 2.1 Download the audio dataset (Requires internet connection)
    python download_real_data.py

    # 2.2 Extract features and apply noise reduction (May take a few minutes)
    python extract_cnn.py

    # 2.3 Train the CNN and save the .h5 model
    python train_cnn.py

3. Setup the Frontend
Open a second terminal window, navigate to the React folder, and install the Node dependencies:

    cd frontend-react
    npm install

################################################
Running the Application
################################################

To use the application, you must run both the backend server and the frontend development server simultaneously.

    Terminal 1 (Backend):
    # From the root ASP_PROJECT directory (ensure 'asp' environment is active)
    python app.py

    (The API will be live at http://127.0.0.1:8000)

    Terminal 2 (Frontend):
    # From the frontend-react directory
    npm run dev

    (The dashboard will be available at http://localhost:5173)

------------------------------------------
Testing the Model
------------------------------------------

Once both servers are running, open your browser to the frontend URL. You can test the system in two ways:

Upload File: Click "Upload MP3/WAV" and select a file from the test_samples/ directory.

Live Mic: Click "Start Live Mic," speak clearly for 3-5 seconds in one of the supported languages, and click "Run Analysis."

For best results, ensure a quiet background environment if not using pre-recorded studio audio.

Developed by: 
Students of Artificial Intelligence and Data Science at Gati Shakti Vishwavidyalaya (GSV)
#Batch 2027