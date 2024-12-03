import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

input_dir = r'C:\Users\sandy\OneDrive\Desktop\dataset\speechocean762\audio' 
output_dir = r'C:\Users\sandy\OneDrive\Desktop\dataset\speechocean762\spectrogram'  

def ensure_output_dir_structure(input_path, output_base):
    relative_path = os.path.relpath(input_path, start=input_dir)
    target_path = os.path.join(output_base, relative_path)
    os.makedirs(target_path, exist_ok=True)
    return target_path

def audio_to_spectrogram(file_path, output_path):
    y, sr = librosa.load(file_path, sr=None)  # Load the audio file
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram of {os.path.basename(file_path)}')
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

# Walk through all subdirectories
total_files = 0
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith('.wav'):  # Check for .wav files (case insensitive)
            file_path = os.path.join(root, file)
            output_folder = ensure_output_dir_structure(root, output_dir)
            output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.png")
            
            audio_to_spectrogram(file_path, output_file)
            print(f"Processed and saved spectrogram for: {file_path}")
            total_files += 1

print(f"Total .wav files processed: {total_files}")
