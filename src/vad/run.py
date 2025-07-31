import webrtcvad
from collections import deque
from typing import Optional, Callable
import time
import asyncio
from src.config.run import logger
from src.config.run import AudioConfig, VADConfig, VoiceState, ConversationState
from src.text.run import ParallelOpenAIHandler

class WebRTCVADProcessor:
    """WebRTC Voice Activity Detection with smart buffering"""
    
    def __init__(self, audio_config: AudioConfig, vad_config: VADConfig):
        self.audio_config = audio_config
        self.vad_config = vad_config
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(vad_config.aggressiveness)
        
        # Voice state tracking
        self.voice_state = VoiceState.SILENCE
        self.voice_frames = 0
        self.silence_frames = 0
        
        # Audio buffering for voice sessions
        self.voice_buffer = deque(maxlen=1000)  # Buffer for voice data
        self.pre_voice_buffer = deque(maxlen=10)  # Small buffer before voice starts
        
        # Timing
        self.frame_duration = self.audio_config.chunk_size / self.audio_config.sample_rate
        self.voice_start_time = None
        
        # Callbacks
        self.on_voice_start: Optional[Callable[[], None]] = None
        self.on_voice_data: Optional[Callable[[bytes], None]] = None
        self.on_voice_end: Optional[Callable[[], None]] = None
        
    def set_callbacks(
        self,
        on_voice_start: Optional[Callable[[], None]] = None,
        on_voice_data: Optional[Callable[[bytes], None]] = None,
        on_voice_end: Optional[Callable[[], None]] = None
    ):
        """Set VAD event callbacks"""
        self.on_voice_start = on_voice_start
        self.on_voice_data = on_voice_data
        self.on_voice_end = on_voice_end
    
    def process_audio_frame(self, audio_data: bytes) -> VoiceState:
        """Process single audio frame through VAD"""
        try:
            # WebRTC VAD requires specific frame size
            if len(audio_data) != self.audio_config.chunk_size * 2:  # 2 bytes per sample for int16
                return self.voice_state
            
            # Run VAD detection
            is_speech = self.vad.is_speech(audio_data, self.audio_config.sample_rate)
            
            if is_speech:
                self.voice_frames += 1
                self.silence_frames = 0
                
                # Always buffer current frame when speech is detected
                self.voice_buffer.append(audio_data)
                
                # Check for voice start
                if (self.voice_state == VoiceState.SILENCE and 
                    self.voice_frames >= self.vad_config.voice_start_threshold):
                    
                    self._start_voice_session()
                
                elif self.voice_state in [VoiceState.VOICE_START, VoiceState.VOICE_ACTIVE]:
                    self.voice_state = VoiceState.VOICE_ACTIVE
                    
                    # Send voice data
                    if self.on_voice_data:
                        self.on_voice_data(audio_data)
            
            else:  # Silence detected
                self.silence_frames += 1
                self.voice_frames = 0
                
                # Always keep some pre-voice context
                if self.voice_state == VoiceState.SILENCE:
                    self.pre_voice_buffer.append(audio_data)
                
                # Check for voice end
                elif (self.voice_state in [VoiceState.VOICE_START, VoiceState.VOICE_ACTIVE] and
                      self.silence_frames >= self.vad_config.voice_end_threshold):
                    
                    self._end_voice_session()
            
            return self.voice_state
            
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            return self.voice_state
    
    def _start_voice_session(self):
        """Start a new voice session"""
        logger.info("🎤 Voice activity started")
        self.voice_state = VoiceState.VOICE_START
        self.voice_start_time = time.time()
        
        # Add pre-voice context to voice buffer
        for frame in self.pre_voice_buffer:
            self.voice_buffer.append(frame)
        
        if self.on_voice_start:
            self.on_voice_start()
        
        # Send buffered pre-voice + current voice data
        if self.on_voice_data:
            for frame in list(self.voice_buffer)[-len(self.pre_voice_buffer)-3:]:
                self.on_voice_data(frame)
    
    def _end_voice_session(self):
        """End current voice session"""
        if self.voice_start_time:
            duration = time.time() - self.voice_start_time
            
            # Only process if voice was long enough
            if duration >= self.vad_config.min_voice_duration:
                logger.info(f"🎤 Voice activity ended (duration: {duration:.2f}s)")
                
                if self.on_voice_end:
                    self.on_voice_end()
            else:
                logger.info("🎤 Voice too short, ignoring")
        
        # Reset state
        self.voice_state = VoiceState.SILENCE
        self.voice_buffer.clear()
        self.voice_start_time = None
        self.voice_frames = 0
        self.silence_frames = 0
    
    def reset(self):
        """Reset VAD state"""
        self.voice_state = VoiceState.SILENCE
        self.voice_frames = 0
        self.silence_frames = 0
        self.voice_buffer.clear()
        self.pre_voice_buffer.clear()
        self.voice_start_time = None
        
    
class ParallelStreamingSpeechConversation:
    """Main class for true parallel streaming conversations with VAD"""
    
    def __init__(
        self,
        audio_config: Optional[AudioConfig] = None,
        vad_config: Optional[VADConfig] = None,
        openai_api_key: Optional[str] = None,
        system_prompt: str = "You are a helpful AI assistant. Keep responses concise and conversational.",
        voice: str = "alloy"
    ):
        self.audio_config = audio_config or AudioConfig()
        self.vad_config = vad_config or VADConfig()
        self.openai_handler = ParallelOpenAIHandler(openai_api_key)
        from src.voice.run import StreamingAudioProcessor
        self.audio_processor = StreamingAudioProcessor(self.audio_config, self.vad_config)
        
        self.system_prompt = system_prompt
        self.voice = voice
        self.state = ConversationState.IDLE
        self.conversation_context = []
        
        # Voice session tracking
        self.current_voice_session = []
        self.voice_session_queue = asyncio.Queue()
        
        # Response tracking
        self.current_response_text = ""
        self.response_sequence = 0
        
        # Callbacks for parallel streams
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_text_chunk: Optional[Callable[[str], None]] = None  # Real-time text
        self.on_text_complete: Optional[Callable[[str], None]] = None  # Complete text
        self.on_audio_chunk: Optional[Callable[[bytes], None]] = None  # Real-time audio
        self.on_audio_complete: Optional[Callable[[], None]] = None  # Audio finished
        self.on_state_change: Optional[Callable[[ConversationState], None]] = None
        self.on_voice_activity: Optional[Callable[[bool], None]] = None  # Voice start/stop
        
        # Control
        self.is_running = False
        
        # Setup VAD callbacks
        self._setup_vad_callbacks()
        
    def _setup_vad_callbacks(self):
        """Setup VAD event callbacks"""
        def on_voice_start():
            self._change_state(ConversationState.VOICE_DETECTED)
            self.current_voice_session = []
            if self.on_voice_activity:
                self.on_voice_activity(True)
        
        def on_voice_data(audio_data: bytes):
            self.current_voice_session.append(audio_data)
        
        def on_voice_end():
            if self.current_voice_session:
                # Queue complete voice session for transcription
                try:
                    self.voice_session_queue.put_nowait(self.current_voice_session.copy())
                except asyncio.QueueFull:
                    logger.warning("Voice session queue full")
            
            self.current_voice_session = []
            self._change_state(ConversationState.LISTENING)
            if self.on_voice_activity:
                self.on_voice_activity(False)
        
        self.audio_processor.set_vad_callbacks(on_voice_start, on_voice_data, on_voice_end)
    
    def set_callbacks(
        self,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_text_chunk: Optional[Callable[[str], None]] = None,
        on_text_complete: Optional[Callable[[str], None]] = None,
        on_audio_chunk: Optional[Callable[[bytes], None]] = None,
        on_audio_complete: Optional[Callable[[], None]] = None,
        on_state_change: Optional[Callable[[ConversationState], None]] = None,
        on_voice_activity: Optional[Callable[[bool], None]] = None
    ):
        """Set callbacks for parallel streaming events"""
        self.on_transcription = on_transcription
        self.on_text_chunk = on_text_chunk
        self.on_text_complete = on_text_complete
        self.on_audio_chunk = on_audio_chunk
        self.on_audio_complete = on_audio_complete
        self.on_state_change = on_state_change
        self.on_voice_activity = on_voice_activity
    
    def _change_state(self, new_state: ConversationState):
        """Change conversation state"""
        self.state = new_state
        logger.info(f"State: {new_state.value}")
        if self.on_state_change:
            self.on_state_change(new_state)
    
    def _text_chunk_callback(self, text_chunk: str):
        """Callback for text chunks (immediate display)"""
        if self.on_text_chunk:
            self.on_text_chunk(text_chunk)
        self.current_response_text += text_chunk
    
    def _audio_chunk_callback(self, audio_data: bytes):
        """Callback for audio chunks (immediate playback)"""
        # Queue audio for immediate playback
        self.audio_processor.queue_audio(audio_data)
        
        if self.on_audio_chunk:
            self.on_audio_chunk(audio_data)
    
    async def _transcription_task(self):
        """Handle VAD-triggered transcription"""
        while self.is_running:
            try:
                # Wait for complete voice session from VAD
                voice_session = await self.voice_session_queue.get()
                
                if not voice_session:
                    continue
                
                # Transcribe the complete voice session
                transcription = await self.openai_handler.transcribe_voice_session(voice_session)
                
                if transcription and transcription.strip():
                    logger.info(f"Transcribed: {transcription}")
                    
                    if self.on_transcription:
                        self.on_transcription(transcription)
                    
                    # Trigger response generation
                    await self._process_user_input(transcription)
                    
            except Exception as e:
                logger.error(f"Transcription task error: {e}")
    
    async def _process_user_input(self, user_text: str):
        """Process user input and generate response"""
        try:
            self._change_state(ConversationState.PROCESSING)
            
            # Add to conversation context
            self.conversation_context.append({"role": "user", "content": user_text})
            
            # Prepare messages
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_context[-10:])
            
            self._change_state(ConversationState.RESPONDING)
            
            # Reset current response
            self.current_response_text = ""
            self.response_sequence += 1
            
            # Generate parallel response (text and audio streams independently)
            complete_response = await self.openai_handler.generate_parallel_response(
                messages=messages,
                text_callback=self._text_chunk_callback,
                audio_callback=self._audio_chunk_callback,
                voice=self.voice
            )
            
            # Add complete response to context
            self.conversation_context.append({
                "role": "assistant", 
                "content": self.current_response_text
            })
            
            # Notify completion
            if self.on_text_complete:
                self.on_text_complete(self.current_response_text)
            
            if self.on_audio_complete:
                self.on_audio_complete()
            
            self._change_state(ConversationState.LISTENING)
            
        except Exception as e:
            logger.error(f"Response processing error: {e}")
            self._change_state(ConversationState.LISTENING)
    
    async def start_conversation(self):
        """Start parallel streaming conversation with VAD"""
        logger.info("Starting VAD-enabled parallel streaming conversation...")
        self.is_running = True
        self._change_state(ConversationState.LISTENING)
        
        # Start audio streams
        self.audio_processor.start_input_stream()
        self.audio_processor.start_output_stream()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._transcription_task())
        ]
        
        try:
            while self.is_running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Conversation interrupted")
        finally:
            self.is_running = False
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.stop_conversation()
    
    async def stop_conversation(self):
        """Stop conversation and cleanup"""
        logger.info("Stopping conversation...")
        self.is_running = False
        self._change_state(ConversationState.IDLE)
        
        self.audio_processor.stop_streams()
        self.audio_processor.cleanup()
    
    def get_conversation_context(self) -> list:
        """Get conversation context"""
        return self.conversation_context.copy()
  
        