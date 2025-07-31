import queue
from src.config.run import AudioConfig, VADConfig, pyaudio
from src.vad.run import WebRTCVADProcessor, Optional, Callable
from src.config.run import logger
import threading
import time

class StreamingAudioProcessor:
    """Handles continuous audio streaming with VAD integration and output interference prevention"""
    
    def __init__(self, config: AudioConfig, vad_config: VADConfig):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.is_streaming = False
        self.input_stream = None
        self.output_stream = None
        self.audio_queue = queue.Queue()
        self.playback_thread = None
        
        # VAD integration
        self.vad_processor = WebRTCVADProcessor(config, vad_config)
        self.voice_session_active = False
        
        # Output tracking for interference prevention
        self.is_audio_playing = False
        self.audio_start_time = None
        self.last_audio_time = None
        
    def set_vad_callbacks(
        self,
        on_voice_start: Optional[Callable[[], None]] = None,
        on_voice_data: Optional[Callable[[bytes], None]] = None,
        on_voice_end: Optional[Callable[[], None]] = None
    ):
        """Set VAD callbacks"""
        self.vad_processor.set_callbacks(on_voice_start, on_voice_data, on_voice_end)
        
    def start_input_stream(self, callback: Optional[Callable[[bytes], None]] = None):
        """Start continuous audio input streaming with VAD"""
        def audio_callback(in_data, frame_count, time_info, status):
            if self.is_streaming:
                # Process through VAD first
                voice_state = self.vad_processor.process_audio_frame(in_data)
                
                # Optional raw audio callback (for debugging)
                if callback:
                    callback(in_data)
            
            return (in_data, pyaudio.paContinue)
        
        self.input_stream = self.audio.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
            stream_callback=audio_callback
        )
        
        self.is_streaming = True
        self.input_stream.start_stream()
        logger.info("🎤 Input stream started with VAD processing")
        
    def start_output_stream(self):
        """Start independent audio output stream with queue-based playback and VAD state management"""
        self.output_stream = self.audio.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            output=True,
            frames_per_buffer=self.config.chunk_size
        )
        
        def audio_playback_worker():
            """Worker thread for continuous audio playback with VAD state management"""
            while self.is_streaming:
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    if audio_data and self.output_stream:
                        # Notify VAD that audio output is starting
                        if not self.is_audio_playing:
                            self.is_audio_playing = True
                            self.audio_start_time = time.time()
                            self.vad_processor.set_output_state(True)
                            logger.debug("🔊 Audio playback started")
                        
                        # Update last audio time
                        self.last_audio_time = time.time()
                        
                        # Play audio
                        self.output_stream.write(audio_data)
                    
                    self.audio_queue.task_done()
                    
                except queue.Empty:
                    # Check if we should stop audio playback state
                    if (self.is_audio_playing and 
                        self.last_audio_time and 
                        time.time() - self.last_audio_time > 0.5):  # 500ms silence threshold
                        
                        self.is_audio_playing = False
                        self.vad_processor.set_output_state(False)
                        logger.debug("🔇 Audio playback stopped")
                    
                    continue
                    
                except Exception as e:
                    logger.error(f"Audio playback error: {e}")
                    # Reset audio state on error
                    if self.is_audio_playing:
                        self.is_audio_playing = False
                        self.vad_processor.set_output_state(False)
        
        self.playback_thread = threading.Thread(target=audio_playback_worker, daemon=True)
        self.playback_thread.start()
        logger.info("🔊 Output stream started with VAD state management")
        
    def queue_audio(self, audio_data: bytes):
        """Queue audio data for immediate playback"""
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            logger.warning("Audio queue full, dropping audio chunk")
    
    def force_stop_audio_output(self):
        """Force stop audio output and notify VAD immediately"""
        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        # Reset audio state
        if self.is_audio_playing:
            self.is_audio_playing = False
            self.vad_processor.set_output_state(False)
            logger.info("🔇 Audio output force stopped")
    
    def is_currently_playing_audio(self) -> bool:
        """Check if audio is currently being played"""
        return self.is_audio_playing
    
    def get_audio_queue_size(self) -> int:
        """Get current audio queue size"""
        return self.audio_queue.qsize()
    
    def stop_streams(self):
        """Stop all audio streams"""
        logger.info("Stopping audio streams...")
        self.is_streaming = False
        
        # Force stop audio output
        self.force_stop_audio_output()
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            logger.info("🎤 Input stream stopped")
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            logger.info("🔊 Output stream stopped")
            
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
            if self.playback_thread.is_alive():
                logger.warning("Playback thread did not stop gracefully")
    
    def cleanup(self):
        """Clean up audio resources"""
        self.stop_streams()
        
        # Reset VAD processor
        self.vad_processor.reset()
        
        # Terminate PyAudio
        self.audio.terminate()
        logger.info("Audio processor cleanup completed")
    
    def get_vad_state(self):
        """Get current VAD state for debugging"""
        return {
            'voice_state': self.vad_processor.voice_state.value,
            'is_output_playing': self.is_audio_playing,
            'output_cooldown': self.vad_processor._is_in_output_cooldown(),
            'should_process_vad': self.vad_processor._should_process_vad(),
            'audio_queue_size': self.get_audio_queue_size()
        }