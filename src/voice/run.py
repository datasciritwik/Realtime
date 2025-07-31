import queue
from src.config.run import AudioConfig, VADConfig, pyaudio
from src.vad.run import WebRTCVADProcessor, Optional, Callable
from src.config.run import logger
import threading

class StreamingAudioProcessor:
    """Handles continuous audio streaming with VAD integration"""
    
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
        
    def start_output_stream(self):
        """Start independent audio output stream with queue-based playback"""
        self.output_stream = self.audio.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            output=True,
            frames_per_buffer=self.config.chunk_size
        )
        
        def audio_playback_worker():
            """Worker thread for continuous audio playback"""
            while self.is_streaming:
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    if audio_data and self.output_stream:
                        self.output_stream.write(audio_data)
                    self.audio_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Audio playback error: {e}")
        
        self.playback_thread = threading.Thread(target=audio_playback_worker, daemon=True)
        self.playback_thread.start()
        
    def queue_audio(self, audio_data: bytes):
        """Queue audio data for immediate playback"""
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            logger.warning("Audio queue full, dropping audio chunk")
    
    def stop_streams(self):
        """Stop all audio streams"""
        self.is_streaming = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
    
    def cleanup(self):
        """Clean up audio resources"""
        self.stop_streams()
        self.audio.terminate()