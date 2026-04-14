from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import librosa
import numpy as np
import joblib
import os
import tempfile
import tensorflow as tf
import noisereduce as nr
import io
import base64
import matplotlib.pyplot as plt
import librosa.display

app = FastAPI(title="CNN Language Vision API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

print("Booting CNN Vision Pipeline...")
model = tf.keras.models.load_model("cnn_language_model.h5")
le = joblib.load("cnn_label_encoder.pkl")
print("System Online.")

def extract_cnn_spectrogram(file_path):
    try:
        # Load the raw audio
        audio, sample_rate = librosa.load(file_path, sr=22050, res_type='kaiser_fast') 

        # 1. Trim extreme silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        if len(audio) == 0: return None
        
        # 2. Normalize volume
        audio = librosa.util.normalize(audio)

        # 3. Force exact length so all images are the same dimensions
        target_length = 5 * sample_rate
        audio = librosa.util.fix_length(audio, size=target_length)
            
        # 4. Generate the pristine spectrogram image
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    except Exception as e:
        print(f"Extraction error: {e}")
        return None

@app.post("/predict/")
async def predict_language(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
        temp_audio.write(await file.read())
        temp_path = temp_audio.name

    try:
        spectrogram = extract_cnn_spectrogram(temp_path)
        if spectrogram is None:
            raise HTTPException(status_code=500, detail="Failed to extract Spectrogram.")

        # --- NEW: Generate the Heatmap Image for the Frontend ---
        plt.figure(figsize=(7, 3))
        # Use the standard 'magma' colormap for a professional look
        librosa.display.specshow(spectrogram, sr=22050, x_axis='time', y_axis='mel', cmap='magma')
        plt.xlabel("Time (Seconds)", fontsize=10, color='#1e293b')
        plt.ylabel("Hz", fontsize=10, color='#1e293b')
        plt.tight_layout(pad=1.0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.close()
        heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        # --------------------------------------------------------

        img_height, img_width = spectrogram.shape
        cnn_input = spectrogram.reshape(1, img_height, img_width, 1)

        probabilities = model.predict(cnn_input)[0]
        prediction_num = np.argmax(probabilities)
        confidence = float(np.max(probabilities) * 100)
        predicted_language = le.inverse_transform([prediction_num])[0]

        distribution = [
            {"language": lang.upper(), "probability": float(prob * 100)}
            for lang, prob in zip(le.classes_, probabilities)
        ]

        return {
            "predicted_language": predicted_language,
            "confidence": f"{confidence:.2f}%",
            "distribution": distribution,
            "heatmap": heatmap_base64  # Send the image to React!
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)