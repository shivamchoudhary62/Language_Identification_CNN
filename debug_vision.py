import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

print("Generating Vision Diagnostic...")

# 1. Load your exact test file
audio, sr = librosa.load("english_human_0000.wav", sr=22050)

# Extract a 5-second chunk from the middle (what the CNN sees)
target_length = 5 * sr
start_index = (len(audio) - target_length) // 2
audio_chunk = audio[start_index : start_index + target_length]

# 2. Generate the Spectrogram Image
mel_spec = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# 3. Save the image so you can look at it
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('What the CNN is actually seeing (hindi.mp3)')
plt.tight_layout()
plt.savefig('debug_spectrogram1.png')

print("Saved to debug_spectrogram.png! Open it and take a look.")
