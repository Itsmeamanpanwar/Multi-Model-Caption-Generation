from app.services.audio_caption import AudioCaptioner


audio_path = "data/audio/test.wav"

captioner = AudioCaptioner()

caption = captioner.generate_caption(audio_path)

print("\nGenerated Audio Caption:")
print(caption)
