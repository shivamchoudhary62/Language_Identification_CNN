import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# We use the Google FLEURS dataset codes for our 5 languages
LANGUAGES = {
    "english": "en_us",
    "french": "fr_fr",
    "german": "de_de",
    "hindi": "hi_in",
    "spanish": "es_419"
}

SAMPLES_PER_LANG = 400

print("Initializing Real Human Speech Download (Google FLEURS)...")

for lang_name, lang_code in LANGUAGES.items():
    save_dir = f"data/{lang_name}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nFetching {lang_name.upper()} speakers...")
    
    # streaming=True means it won't download the massive 100GB dataset, 
    # it just streams exactly the 200 clips we need!
    try:
        dataset = load_dataset("google/fleurs", lang_code, split="train", streaming=True, trust_remote_code=True)
        
        count = 0
        # Wrap the dataset in tqdm for a progress bar
        pbar = tqdm(total=SAMPLES_PER_LANG, desc=f"Saving {lang_name} .wav files")
        
        for item in dataset:
            if count >= SAMPLES_PER_LANG:
                break
                
            # Extract the raw audio array and the sample rate from Hugging Face
            audio_data = item["audio"]["array"]
            sample_rate = item["audio"]["sampling_rate"]
            
            # Save it as a standard .wav file
            file_path = os.path.join(save_dir, f"{lang_name}_human_{count:04d}.wav")
            sf.write(file_path, audio_data, sample_rate)
            
            count += 1
            pbar.update(1)
            
        pbar.close()
    except Exception as e:
        print(f"Error downloading {lang_name}: {e}")

print("\nDataset generation complete! You now have 2,000 real human audio clips.")