import os
import requests
import speech_recognition as sr
from pydub import AudioSegment
import subprocess

def download_m3u8_from_url(m3u8_url, output_m3u8):
    # Download the m3u8 file from the URL
    response = requests.get(m3u8_url)
    
    if response.status_code == 200:
        with open(output_m3u8, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download m3u8 file. HTTP status code: {response.status_code}")
    
def convert_m3u8_to_wav(m3u8_url, output_wav_file):
    subprocess.run(['ffmpeg', '-i', m3u8_url, output_wav_file])

def speech_to_text(wav_file):
    # Initialize the SpeechRecognition recognizer
    recognizer = sr.Recognizer()

    # Use the recognize_google method for Google Web Speech API
    with sr.WavFile(wav_file) as source:
        audio_data = recognizer.record(source)
        # Recognize speech using Google Web Speech API
        try:
            text = recognizer.recognize_google(audio_data, language='th')
            return text
        except sr.UnknownValueError:
            return "Speech recognition could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Web Speech API; {e}"
        
def split_wav(wav_file, segment_duration=10):
    # Load the entire audio file
    audio = AudioSegment.from_wav(wav_file)

    # Get the total duration of the audio in seconds
    total_duration = len(audio) / 1000.0

    # Calculate the number of segments needed
    num_segments = int(total_duration / segment_duration) + 1

    # Split the audio into segments
    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration * 1000
        end_time = min((i + 1) * segment_duration * 1000, len(audio))
        segment = audio[start_time:end_time]
        segments.append(segment)

    return segments

def speech_to_text_segments(segments, temp_wav_file):
    results = []

    for index, segment in enumerate(segments):
        segment.export(temp_wav_file, format="wav")

        # Use the speech_to_text function on the temporary WAV file
        result = speech_to_text(temp_wav_file)
        results.append(result)

        # Remove the temporary WAV file
        os.remove(temp_wav_file)
        print(f"[S2T] sengment {index + 1}/{len(segments)}")

    return results

def concatenate_results(results):
    return ' '.join(results)

def execute_speech_recognition(wav_file, temp_wav_file):
    segment_duration = 10  # in seconds

    # Split the WAV file into segments
    segments = split_wav(wav_file, segment_duration)

    # Perform speech recognition on each segment
    results = speech_to_text_segments(segments, temp_wav_file)

    # Concatenate the results
    final_result = concatenate_results(results)
    return final_result
