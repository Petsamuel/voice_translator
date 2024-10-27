import pandas as pd
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
import warnings
import os
import tqdm
import unidecode
from googletrans import Translator
import sounddevice as sd
import queue
import threading
from gtts import gTTS
import pygame
import time

class VoiceTranslator:
    def __init__(self, model_path=None, sample_rate=16000):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.translator = Translator()
        self.running = False
        self.trained = False
        
        # Initialize feature extraction parameters
        self.n_mfcc = 40
        self.hop_length = 512
        self.n_fft = 2048
        
        # Initialize classifier and labels
        self.clf = None
        self.labels = None
        
        pygame.mixer.init()

    def feature_extractor(self, audio_data):
        """Extract MFCC features from audio data"""
        warnings.filterwarnings('ignore')
        try:
            # Ensure audio_data is the correct type and shape
            audio_data = np.array(audio_data, dtype=np.float32)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert stereo to mono if needed
            
            # Pad or truncate audio to ensure consistent length
            target_length = 3 * self.sample_rate  # 3 seconds of audio
            if len(audio_data) > target_length:
                audio_data = audio_data[:target_length]
            else:
                audio_data = np.pad(audio_data, (0, max(0, target_length - len(audio_data))))

            # Extract MFCC features
            feat = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # Calculate statistics over time
            features = np.concatenate([
                feat.mean(axis=1),
                feat.std(axis=1),
                feat.min(axis=1),
                feat.max(axis=1)
            ])
            
            return features.reshape(-1)  # Ensure 1D array
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return None
        finally:
            warnings.filterwarnings("default")

    def train_model(self, dataset_dir):
        """Train the model on a dataset"""
        print("Starting model training...")
        
        # Validate directory and TSV file
        tsv_path = os.path.join(dataset_dir, 'other.tsv') 
        if not os.path.exists(tsv_path):
            raise ValueError(f"TSV file not found at {tsv_path}")
        
        # Read TSV file with explicit encoding
        try:
            df = pd.read_csv(tsv_path, delimiter="\t", encoding='utf-8')
            print(f"Found {len(df)} entries in TSV file")
        except Exception as e:
            print(f"Error reading TSV file: {e}")
            return

        # Prepare for feature extraction
        X = []
        y = []
        clips_dir = os.path.join(dataset_dir, "clips")
        
        # Process each audio file with detailed logging
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
            try:
                # Construct full audio path
                audio_path = os.path.join(clips_dir, str(row['path']))
                
                if not os.path.exists(audio_path):
                    print(f"Skipping missing file: {audio_path}")
                    continue
                
                # Load and process audio
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=3)
                
                # Extract features
                features = self.feature_extractor(audio)
                if features is not None:
                    X.append(features)
                    y.append(str(row['sentence']))  # Store full sentence without normalization
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

        # Validate processed data
        if len(X) == 0:
            raise ValueError("No valid audio files were processed")
        
        print(f"\nSuccessfully processed {len(X)} files")
        
        # Convert to numpy arrays with explicit types
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Train KNN classifier
        self.clf = KNeighborsClassifier(n_neighbors=3)  # Reduced neighbors for small dataset
        self.clf.fit(X, y)
        self.labels = y
        
        self.trained = True
        print("Model training complete!")

    def process_audio(self):
        """Process audio chunks from the queue"""
        if not self.trained:
            raise ValueError("Model must be trained before processing audio")
        
        # Buffer for collecting audio chunks
        audio_buffer = []
        
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                audio_buffer.append(audio_data.flatten())
                
                # Process when we have enough audio (3 seconds)
                if len(audio_buffer) >= 3 * (self.sample_rate // self.hop_length):
                    # Combine buffer chunks
                    combined_audio = np.concatenate(audio_buffer)
                    features = self.feature_extractor(combined_audio)
                    
                    if features is not None:
                        # Make prediction
                        prediction = self.clf.predict([features])[0]
                        
                        # Translate
                        try:
                            translation = self.translator.translate(prediction, dest='en')
                            print(f"\nOriginal: {prediction}")
                            print(f"Translation: {translation.text}")
                            
                            # Text to speech
                            tts = gTTS(text=translation.text, lang='en')
                            temp_file = "output_temp.mp3"
                            tts.save(temp_file)
                            
                            # Play audio
                            pygame.mixer.music.load(temp_file)
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy():
                                time.sleep(0.1)
                            
                            # Cleanup
                            try:
                                os.remove(temp_file)
                            except:
                                pass
                        except Exception as e:
                            print(f"Error in translation/playback: {e}")
                    
                    # Reset buffer
                    audio_buffer = []
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
                audio_buffer = []  # Reset buffer on error

    def start(self, duration=None):
        """Start real-time translation"""
        if not self.trained:
            raise ValueError("Model must be trained before starting translation")
            
        self.running = True
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_audio)
        processing_thread.start()
        
        # Calculate buffer size in samples
        buffer_size = self.sample_rate // 10  # 100ms buffer
        
        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=buffer_size
            ):
                print("\nStarted recording... (Press Ctrl+C to stop)")
                if duration:
                    time.sleep(duration)
                else:
                    while self.running:
                        time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.running = False
            processing_thread.join()

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def stop(self):
        """Stop the translation system"""
        self.running = False

if __name__ == "__main__":
    # Example usage
    translator = VoiceTranslator()
    
    try:
        # Update this path to your dataset directory
        dataset_path = "./lg"
        translator.train_model(dataset_path)
        
        # Start real-time translation
        translator.start()
    except KeyboardInterrupt:
        translator.stop()
    except Exception as e:
        print(f"Error: {e}")