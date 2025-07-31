from openai import AsyncOpenAI
import asyncio
from typing import Optional, Callable
from src.config.run import logger
import os


class ParallelOpenAIHandler:
    """Handles parallel OpenAI API interactions with VAD-processed audio"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
    async def transcribe_voice_session(self, audio_chunks: list[bytes]) -> Optional[str]:
        """Transcribe a complete voice session from VAD"""
        try:
            if not audio_chunks:
                return None
            
            # Combine all audio chunks
            combined_audio = b''.join(audio_chunks)
            
            # Convert to WAV format for Whisper
            import io
            import wave
            
            audio_io = io.BytesIO()
            with wave.open(audio_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)
                wav_file.writeframes(combined_audio)
            
            audio_io.seek(0)
            audio_io.name = "voice_session.wav"
            
            response = await self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_io,
                response_format="text"
            )
            
            text = response.strip()
            return text if text else None
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    async def generate_parallel_response(
        self, 
        messages: list, 
        text_callback: Callable[[str], None],
        audio_callback: Callable[[bytes], None],
        voice: str = "alloy"
    ):
        """Generate text and audio in parallel streams"""
        
        # Text generation task
        async def text_stream_task():
            try:
                full_text = ""
                stream = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7,
                    stream=True
                )
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        text_chunk = chunk.choices[0].delta.content
                        full_text += text_chunk
                        
                        # Immediate text callback
                        text_callback(text_chunk)
                
                return full_text
                
            except Exception as e:
                logger.error(f"Text generation error: {e}")
                return "I'm sorry, I couldn't process that."
        
        # Audio generation task (runs in parallel)
        async def audio_stream_task(response_text_future):
            try:
                # Wait for some initial text before starting TTS
                await asyncio.sleep(0.5)
                
                # We'll collect text in sentences and convert to audio
                sentence_buffer = ""
                sentence_endings = '.!?'
                processed_length = 0
                
                while True:
                    try:
                        # Get the current full text
                        if response_text_future.done():
                            full_text = response_text_future.result()
                            remaining_text = full_text[processed_length:]
                            
                            if remaining_text:
                                sentence_buffer += remaining_text
                                processed_length = len(full_text)
                        else:
                            # Still generating, wait a bit
                            await asyncio.sleep(0.1)
                            continue
                        
                        # Process complete sentences
                        while any(ending in sentence_buffer for ending in sentence_endings):
                            earliest_pos = len(sentence_buffer)
                            for ending in sentence_endings:
                                pos = sentence_buffer.find(ending)
                                if pos != -1 and pos < earliest_pos:
                                    earliest_pos = pos
                            
                            if earliest_pos < len(sentence_buffer):
                                sentence = sentence_buffer[:earliest_pos + 1].strip()
                                sentence_buffer = sentence_buffer[earliest_pos + 1:].strip()
                                
                                if sentence:
                                    # Convert to audio (this runs independently)
                                    audio_data = await self._text_to_speech_chunk(sentence, voice)
                                    if audio_data:
                                        # Resample from 24kHz to 16kHz for output consistency
                                        resampled_audio = self._resample_audio(audio_data, 24000, 16000)
                                        audio_callback(resampled_audio)
                        
                        # If text generation is complete and no more sentences, process remaining
                        if response_text_future.done() and sentence_buffer.strip():
                            audio_data = await self._text_to_speech_chunk(sentence_buffer.strip(), voice)
                            if audio_data:
                                resampled_audio = self._resample_audio(audio_data, 24000, 16000)
                                audio_callback(resampled_audio)
                            break
                            
                        # If text generation is complete and no remaining text, break
                        if response_text_future.done() and not sentence_buffer.strip():
                            break
                            
                    except Exception as e:
                        logger.error(f"Audio processing error: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"Audio generation error: {e}")
        
        # Run both tasks in parallel
        text_task = asyncio.create_task(text_stream_task())
        audio_task = asyncio.create_task(audio_stream_task(text_task))
        
        # Wait for both to complete
        results = await asyncio.gather(text_task, audio_task, return_exceptions=True)
        
        return results[0] if not isinstance(results[0], Exception) else "Error generating response"
    
    async def _text_to_speech_chunk(self, text: str, voice: str) -> Optional[bytes]:
        """Convert a text chunk to speech"""
        try:
            response = await self.client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format="wav"
            )
            return response.content
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def _resample_audio(self, audio_data: bytes, source_rate: int, target_rate: int) -> bytes:
        """Simple audio resampling (basic implementation)"""
        if source_rate == target_rate:
            return audio_data
        
        try:
            # Extract WAV data (skip header)
            import wave
            import io
            
            # Read WAV data
            wav_io = io.BytesIO(audio_data)
            with wave.open(wav_io, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
            
            # Convert to numpy for resampling
            import numpy as np
            audio_array = np.frombuffer(frames, dtype=np.int16)
            
            # Simple decimation/interpolation
            ratio = target_rate / source_rate
            target_length = int(len(audio_array) * ratio)
            
            # Linear interpolation
            resampled = np.interp(
                np.linspace(0, len(audio_array), target_length),
                np.arange(len(audio_array)),
                audio_array
            ).astype(np.int16)
            
            return resampled.tobytes()
            
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return audio_data  # Return original if resampling fails
