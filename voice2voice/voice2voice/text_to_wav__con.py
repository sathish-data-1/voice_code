# import pandas as pd
# import os
# import pyttsx3

# # Load CSV
# df = pd.read_csv("Conversation.csv")

# # Output directory
# output_dir = os.path.join(os.getcwd(), "data", "raw")
# os.makedirs(output_dir, exist_ok=True)

# # Initialize the text-to-speech engine
# engine = pyttsx3.init()

# # Optional: Set voice (e.g., male/female)
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[0].id)  # 0 for male, 1 for female (varies by system)

# # Optional: Set speaking rate (default is ~200 wpm)
# engine.setProperty('rate', 160)

# def text_to_wav_pyttsx3(text, filename):
#     path = os.path.join(output_dir, f"{filename}.wav")
#     engine.save_to_file(text, path)
#     engine.runAndWait()

# # Convert each question and answer
# for idx, row in df.iterrows():
#     question = row['question']
#     answer = row['answer']
    
#     q_filename = f"input_{idx:03d}"
#     a_filename = f"output_{idx:03d}"
    
#     text_to_wav_pyttsx3(question, q_filename)
#     text_to_wav_pyttsx3(answer, a_filename)

# print(f"✅ Conversion done using pyttsx3! Check folder: {output_dir}")


import pandas as pd
import os
import pyttsx3

# Load CSV
df = pd.read_csv("Conversation.csv")

# Output directories
output_dir = os.path.join(os.getcwd(), "data", "raw")
os.makedirs(output_dir, exist_ok=True)

# Path for train_files.txt
train_file_path = os.path.join(output_dir, "train_files.txt")

# Initialize pyttsx3
engine = pyttsx3.init()

# Optional: Select voice (0 for male, 1 for female, varies by system)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# Optional: Adjust speech rate
engine.setProperty('rate', 160)

def text_to_wav(text, filename):
    path = os.path.join(output_dir, f"{filename}.wav")
    engine.save_to_file(text, path)

# Collect mappings for train_files.txt
mappings = []

# Loop through rows to generate .wav files
for idx, row in df.iterrows():
    question = row['question']
    answer = row['answer']

    input_filename = f"input_{idx:03d}"
    output_filename = f"output_{idx:03d}"

    text_to_wav(question, input_filename)
    text_to_wav(answer, output_filename)

    mappings.append(f"{input_filename}.wav|{output_filename}.wav")

# Run TTS engine to save all files
engine.runAndWait()

# Write mappings to train_files.txt
with open(train_file_path, "w", encoding='utf-8') as f:
    f.write("text\n")
    for line in mappings:
        f.write(line + "\n")

print(f"✅ All audio files and train_files.txt saved in: {output_dir}")
