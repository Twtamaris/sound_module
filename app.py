from pydub import AudioSegment
import os

# Specify the location of ffmpeg
AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"
print((AudioSegment.converter))

# Load the audio file
audio = AudioSegment.from_file("sound/sound.mp3")

# Create a directory to store the clips if it doesn't exist
output_directory = "sound_folder"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Length of each clip in milliseconds (0.1 second = 100 milliseconds)
clip_length = 100

print(len(audio))

# Split the audio into 0.1-second clips
for i in range(0, 55920, clip_length):
    clip = audio[i:i + clip_length]
    clip.export(f"{output_directory}/clip_{i // clip_length}.wav", format="wav")
