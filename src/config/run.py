import logging
from dataclasses import dataclass
import pyaudio
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    VOICE_DETECTED = "voice_detected"
    PROCESSING = "processing"
    RESPONDING = "responding"  # Both text and audio are being generated


class VoiceState(Enum):
    SILENCE = "silence"
    VOICE_START = "voice_start"
    VOICE_ACTIVE = "voice_active"
    VOICE_END = "voice_end"


@dataclass
class AudioConfig:
    """Audio configuration settings optimized for WebRTC VAD"""
    sample_rate: int = 16000  # WebRTC VAD requires 8kHz, 16kHz, 32kHz, or 48kHz
    chunk_size: int = 320     # 20ms at 16kHz (320 samples) - WebRTC VAD requirement
    channels: int = 1
    format: int = pyaudio.paInt16


@dataclass
class VADConfig:
    """Voice Activity Detection configuration"""
    aggressiveness: int = 2          # 0-3, higher = more aggressive filtering
    min_voice_duration: float = 0.3  # Minimum voice duration to trigger (seconds)
    max_silence_duration: float = 1.5 # Max silence before ending voice session (seconds)
    voice_start_threshold: int = 3    # Consecutive voice frames to start
    voice_end_threshold: int = 10     # Consecutive silence frames to end


@dataclass
class ResponseChunk:
    """Represents a chunk of response (text or audio)"""
    type: str  # 'text' or 'audio'
    content: Any  # str for text, bytes for audio
    timestamp: float
    sequence_id: int