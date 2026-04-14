import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_PATH = "data/"
DURATION = 5 # Force all clips to 5 seconds
SAMPLE_RATE = 22050

def get_spectrogram(audio):
    # 1. Force exact length so all images are the same dimensions
    audio = librosa.util.fix_length(audio, size=DURATION * SAMPLE_RATE)
    
    # 2. Generate the Mel-Spectrogram (The visual heat-map of the audio)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
    
    # 3. Convert to Decibels (Log scale is required for Neural Networks)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

extracted_features = []

for language_folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, language_folder)
    if os.path.isdir(folder_path):
        print(f"Processing Spectrograms for {language_folder}...")
        
        for file_name in tqdm(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)
            try:
                audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, res_type='kaiser_fast')
                audio, _ = librosa.effects.trim(audio, top_db=20)
                audio = librosa.util.normalize(audio)

                # Extract Clean Spectrogram
                spec = get_spectrogram(audio)
                extracted_features.append([spec, language_folder])

                # Data Augmentation: Extract Noisy Spectrogram
                noise = np.random.randn(len(audio))
                noisy_audio = audio + 0.005 * noise
                noisy_spec = get_spectrogram(noisy_audio)
                extracted_features.append([noisy_spec, language_folder])
                
            except Exception as e:
                pass

df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
df.to_pickle("cnn_audio_features.pkl")
print("\nSuccess! Saved to cnn_audio_features.pkl")