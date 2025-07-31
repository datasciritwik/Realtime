from src.config.run import VADConfig
from src.vad.run import ParallelStreamingSpeechConversation
from src.config.run import ConversationState
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def main():
    """Example VAD-enabled parallel streaming conversation"""
    
    # Configure VAD sensitivity
    vad_config = VADConfig(
        aggressiveness=2,              # 0=least aggressive, 3=most aggressive
        min_voice_duration=0.5,        # Minimum 0.5s of voice to trigger
        max_silence_duration=1.0,      # 1s of silence ends voice session
        voice_start_threshold=3,       # 3 consecutive voice frames to start
        voice_end_threshold=8          # 8 consecutive silence frames to end
    )
    
    conversation = ParallelStreamingSpeechConversation(
        vad_config=vad_config,
        system_prompt="You are a friendly AI. Keep responses brief and natural.",
        voice="alloy"
    )
    
    # Track display state
    current_text_line = ""
    
    def on_voice_activity(is_active: bool):
        if is_active:
            print("\n🎤 Voice detected, listening...")
        else:
            print("🎤 Voice ended, processing...")
    
    def on_transcription(text: str):
        print(f"\n💬 You said: {text}")
        print("🤖 AI: ", end='', flush=True)
    
    def on_text_chunk(chunk: str):
        """Display text immediately as it generates"""
        nonlocal current_text_line
        print(chunk, end='', flush=True)
        current_text_line += chunk
    
    def on_text_complete(full_text: str):
        """Text generation complete"""
        print(f" ✓")
    
    def on_audio_chunk(audio_data: bytes):
        """Audio is playing (visual indicator)"""
        pass  # Audio plays automatically through the queue
    
    def on_audio_complete():
        """Audio playback complete"""
        print(" 🔊✓")
        print("\n👂 Listening for voice...")
    
    def on_state_change(state: ConversationState):
        state_emoji = {
            ConversationState.IDLE: "⏸️",
            ConversationState.LISTENING: "👂",
            ConversationState.VOICE_DETECTED: "🎙️",
            ConversationState.PROCESSING: "🧠",
            ConversationState.RESPONDING: "🗣️"
        }
        if state == ConversationState.LISTENING:
            pass  # Don't spam listening messages
        elif state == ConversationState.VOICE_DETECTED:
            pass  # Voice activity callback handles this
    
    conversation.set_callbacks(
        on_voice_activity=on_voice_activity,
        on_transcription=on_transcription,
        on_text_chunk=on_text_chunk,
        on_text_complete=on_text_complete,
        on_audio_chunk=on_audio_chunk,
        on_audio_complete=on_audio_complete,
        on_state_change=on_state_change
    )
    
    print("🎙️ VAD-Enhanced Parallel Streaming Speech Conversation")
    print("✨ Smart voice detection prevents false triggering!")
    print("🤐 Only processes speech, ignores silence and noise")
    print("💬 Text streams as GPT thinks")
    print("🔊 Audio streams as TTS converts")
    print("⚙️  VAD Settings:")
    print(f"   - Aggressiveness: {vad_config.aggressiveness}/3")
    print(f"   - Min voice duration: {vad_config.min_voice_duration}s")
    print(f"   - Max silence: {vad_config.max_silence_duration}s")
    print("⌨️  Press Ctrl+C to stop\n")
    print("👂 Ready to listen for voice...")
    
    try:
        await conversation.start_conversation()
    except KeyboardInterrupt:
        print("\n👋 Conversation ended")


if __name__ == "__main__":
    asyncio.run(main())