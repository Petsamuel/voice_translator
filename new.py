import pandas as pd
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import os
import tqdm
from googletrans import Translator
import sounddevice as sd
import queue
import threading
from gtts import gTTS
import pygame
import time
from pathlib import Path
from datetime import datetime

class VoiceTranslator:
    def __init__(self, model_path=None, sample_rate=16000):
        self.log_timestamp("Initializing Voice Translator")
        
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue(maxsize=50)  # Reduced queue size
        self.translator = Translator()
        self.running = False
        self.trained = False
        
        # Create temporary directory in the root directory
        self.temp_dir = Path.cwd() / 'voice_translator_temp'
        self.temp_dir.mkdir(exist_ok=True)
        print(f"Using directory: {self.temp_dir}")
        
        # Initialize feature extraction parameters
        self.n_mfcc = 13  
        self.hop_length = 160  
        self.n_fft = 512  
        self.confidence_threshold = 0.05
        self.prediction_history = []
        self.history_size = 10
        
        # Shorter audio processing window
        self.window_duration = 1.5  # Reduced to 1.5 seconds
        
        # Initialize classifier, scaler and labels
        self.clf = None
        self.scaler = StandardScaler()
        self.labels = None
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        self.log_timestamp("Initialization complete")

    def log_timestamp(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[{timestamp}] {message}")

    def feature_extractor(self, audio_data):
        """Simplified feature extraction for faster processing"""
        try:
            # Basic preprocessing
            audio_data = np.array(audio_data, dtype=np.float32)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Simple normalization
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # Simple statistics
            features = np.concatenate([
                mfcc.mean(axis=1),
                mfcc.std(axis=1)
            ])
            
            return features.reshape(-1)
            
        except Exception as e:
            self.log_timestamp(f"Error in feature extraction: {e}")
            return None

    def train_model(self, dataset_dir):
        """Train the model with timing information"""
        self.log_timestamp("Starting model training")
        
        tsv_path = os.path.join(dataset_dir, 'other.tsv')
        if not os.path.exists(tsv_path):
            raise ValueError(f"TSV file not found at {tsv_path}")
        
        try:
            df = pd.read_csv(tsv_path, delimiter="\t", encoding='utf-8')
            self.log_timestamp(f"Found {len(df)} entries in TSV file")
        except Exception as e:
            self.log_timestamp(f"Error reading TSV file: {e}")
            return

        X = []
        y = []
        clips_dir = os.path.join(dataset_dir, "clips")
        
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
            try:
                audio_path = os.path.join(clips_dir, str(row['path']))
                
                if not os.path.exists(audio_path):
                    continue
                
                # Load audio with shorter duration
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.window_duration)
                
                features = self.feature_extractor(audio)
                if features is not None:
                    X.append(features)
                    y.append(str(row['sentence']))
                
            except Exception as e:
                self.log_timestamp(f"Error processing {audio_path}: {e}")
                continue

        if len(X) == 0:
            raise ValueError("No valid audio files were processed")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train KNN classifier
        self.clf = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='kd_tree')
        self.clf.fit(X_scaled, y)
        self.labels = y
        
        self.trained = True
        self.log_timestamp("Model training complete")

    def process_audio(self):
        """Process audio with improved classification and confidence checking"""
        if not self.trained:
            raise ValueError("Model must be trained before processing audio")
        
        audio_buffer = []
        last_prediction = None
        last_translation_time = 0
        min_time_between_translations = 0.5
        
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.05)
                audio_buffer.append(audio_data.flatten())
                
                current_time = time.time()
                buffer_duration = len(audio_buffer) * (self.hop_length / self.sample_rate)
                
                if buffer_duration >= self.window_duration:
                    process_start = time.time()
                    self.log_timestamp("Starting audio processing")
                    
                    combined_audio = np.concatenate(audio_buffer)
                    features = self.feature_extractor(combined_audio)
                    
                    if features is not None:
                        features_scaled = self.scaler.transform([features])
                        
                        # Get distances to k nearest neighbors
                        distances, indices = self.clf.kneighbors(features_scaled)
                        
                        # Calculate confidence score (inverse of mean distance)
                        confidence = 1 / (1 + np.mean(distances))
                        
                        # Get the predicted class
                        prediction = self.clf.predict(features_scaled)[0]
                        
                        # Check confidence threshold and prediction history
                        if (confidence > self.confidence_threshold and 
                            prediction != last_prediction and 
                            current_time - last_translation_time > min_time_between_translations and
                            not self._is_recent_prediction(prediction)):
                            self.log_timestamp(f"\nconfidence_threshold: {self.confidence_threshold}")
                            self.log_timestamp(f"Confidence: {confidence:.3f}")
                            self.log_timestamp("Making translation")
                            
                            try:
                                translation = self.translator.translate(prediction, dest='en')
                                print(f"\nOriginal: {prediction}")
                                print(f"Translation: {translation.text}")
                                
                                # Update prediction history
                                self._update_prediction_history(prediction)
                                
                                temp_file = self.temp_dir / f"output_{int(current_time)}.mp3"
                                
                                tts = gTTS(text=translation.text, lang='en')
                                tts.save(str(temp_file))
                                
                                pygame.mixer.music.load(str(temp_file))
                                pygame.mixer.music.play()
                                
                                last_prediction = prediction
                                last_translation_time = current_time
                                
                                threading.Thread(target=self.cleanup_temp_files, daemon=True).start()
                                
                            except Exception as e:
                                self.log_timestamp(f"Error in translation/playback: {e}")
                        else:
                            self.log_timestamp(f"Prediction rejected (confidence: {confidence:.3f})")
                    
                    process_end = time.time()
                    self.log_timestamp(f"Processing completed in {(process_end - process_start):.3f} seconds")
                    audio_buffer = []
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.log_timestamp(f"Error in audio processing: {e}")
                audio_buffer = []
    def _update_prediction_history(self, prediction):
        """Keep track of recent predictions"""
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)

    def _is_recent_prediction(self, prediction):
        """Check if prediction was made recently"""
        return prediction in self.prediction_history

    def start(self):
        """Start translation with timing information"""
        if not self.trained:
            raise ValueError("Model must be trained before starting translation")
            
        self.running = True
        self.log_timestamp("Starting voice translator")
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_audio)
        processing_thread.daemon = True
        processing_thread.start()
        
        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.05)  # 50ms blocks
            ):
                self.log_timestamp("Started recording")
                while self.running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            self.log_timestamp("Stopping...")
        except Exception as e:
            self.log_timestamp(f"Error in audio stream: {e}")
        finally:
            self.stop()

    def audio_callback(self, indata, frames, time, status):
        """Audio callback with overflow handling"""
        if status:
            self.log_timestamp(f"Status: {status}")
        try:
            self.audio_queue.put(indata.copy(), block=False)
        except queue.Full:
            pass  # Silently drop frames if queue is full

    def cleanup_temp_files(self):
        """Asynchronous cleanup of temporary files"""
        try:
            current_time = time.time()
            for file in self.temp_dir.glob("*.mp3"):
                if current_time - file.stat().st_mtime > 300:  # Remove files older than 5 minutes
                    try:
                        file.unlink()
                    except:
                        pass
        except Exception as e:
            self.log_timestamp(f"Error cleaning up temp files: {e}")

    def stop(self):
        """Stop translation and cleanup"""
        self.running = False
        self.log_timestamp("Stopping voice translator")
        try:
            for file in self.temp_dir.glob("*.mp3"):
                try:
                    file.unlink()
                except:
                    pass
            self.temp_dir.rmdir()
        except Exception as e:
            self.log_timestamp(f"Error cleaning up: {e}")

if __name__ == "__main__":
    translator = VoiceTranslator()
    
    try:
        dataset_path = "./lg"
        translator.train_model(dataset_path)
        translator.start()
    except KeyboardInterrupt:
        translator.stop()
    except Exception as e:
        print(f"Error: {e}")