import pyaudio
import numpy as np
import librosa
import tensorflow as tf

# Load the saved model
loaded_model = tf.keras.models.load_model('sound_detection_module.h5')

# Function to extract MFCC features from audio
def extract_features(audio):
    # Normalize audio data to floating-point values between -1 and 1
    audio_norm = audio.astype(np.float32) / 32768.0  # Assuming audio is in np.int16 format
    mfccs = librosa.feature.mfcc(y=audio_norm, sr=22050, n_mfcc=13)
    return mfccs.T

# Constants
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Number of audio channels
RATE = 22050  # Sampling rate
CHUNK = int(0.1 * RATE)  # Chunk size (0.1 second)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open a stream to capture audio
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Listening...")

while True:
    try:
        # Read audio data from the stream
        audio_data = stream.read(CHUNK)
        
        # Convert audio data to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Extract MFCC features
        features = extract_features(audio_array)
        
        # Reshape and normalize features for model prediction
        features_cnn = features[np.newaxis, ..., np.newaxis]
        
        # Make predictions
        prediction = loaded_model.predict(features_cnn)
        
        # Print the prediction result (e.g., sound or no sound)
        if prediction > 0.5:
            print("Sound detected")
        else:
            print("No sound")
    
    except KeyboardInterrupt:
        break

# Stop and close the audio stream
stream.stop_stream()
stream.close()
audio.terminate()
