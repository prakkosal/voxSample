from vastai import Deployment
from vastai.data.query import gpu_name, RTX_4090, RTX_5090

app = Deployment(name="voxcpm2-tts")

MODEL_NAME = "openbmb/VoxCPM2"
# deploy-bust: hex-bytes-v1

@app.context()
class VoxCPM2Engine:
    async def __aenter__(self):
        from voxcpm import VoxCPM
        import torch
        
        # Load VoxCPM2 model
        # Requires GPU with ~8GB VRAM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VoxCPM.from_pretrained(
            MODEL_NAME,
            device=device,
            load_denoiser=True  # Optional: enables denoiser for better quality
        )
        self.sample_rate = self.model.tts_model.sample_rate
        return self

    async def __aexit__(self, *exc):
        # Cleanup
        if hasattr(self, 'model'):
            del self.model

@app.remote(benchmark_dataset=[{"text": "Hello, this is a test."}])
async def generate_speech(
    text: str,
    reference_wav_hex: str = "",
    voice_description: str = "",
    cfg_value: float = 2.0,
    inference_timesteps: int = 25
) -> dict:
    """
    Generate speech from text using VoxCPM2.

    Args:
        text: The text to convert to speech
        reference_wav_hex: Optional hex-encoded WAV bytes for voice cloning
        voice_description: Optional voice description (e.g., "A young woman with a gentle voice")
        cfg_value: Classifier-free guidance strength (default: 2.0)
        inference_timesteps: Number of inference timesteps (default: 25)

    Returns:
        Dictionary containing generated audio as bytes and metadata
    """
    import uuid
    import os
    import tempfile
    import io

    engine = app.get_context(VoxCPM2Engine)

    tmp_ref_path = None
    try:
        # Handle voice description in text format
        # Format: "(description)actual text to synthesize"
        if voice_description and not text.startswith("("):
            text = f"({voice_description}){text}"

        # Generate speech
        if reference_wav_hex:
            # Voice cloning mode — persist uploaded bytes to a temp file
            # because the underlying model expects a filesystem path.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(bytes.fromhex(reference_wav_hex))
                tmp_ref_path = tmp.name
            wav = engine.model.generate(
                text=text,
                reference_wav_path=tmp_ref_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps
            )
        else:
            # Standard TTS or voice design mode
            wav = engine.model.generate(
                text=text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps
            )
        
        # Convert audio to bytes for transmission
        audio_bytes = io.BytesIO()
        
        # Import soundfile for audio writing
        import soundfile as sf
        sf.write(audio_bytes, wav, engine.sample_rate, format='WAV')
        audio_bytes.seek(0)
        audio_data = audio_bytes.getvalue()
        
        return {
            "status": "success",
            "audio": audio_data.hex(),  # Convert bytes to hex string for JSON serialization
            "sample_rate": engine.sample_rate,
            "duration_seconds": len(wav) / engine.sample_rate,
            "request_id": str(uuid.uuid4())
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "request_id": str(uuid.uuid4())
        }
    finally:
        if tmp_ref_path and os.path.exists(tmp_ref_path):
            os.unlink(tmp_ref_path)

# Container image and dependencies
image = app.image("vastai/vllm:v0.11.0-cuda-12.8-mvc-cuda-12.0", 40)
image.use_system_python()

# Install VoxCPM2 and dependencies
image.pip_install(
    "voxcpm>=0.1.0",
    "soundfile>=0.12.0",
    "transformers>=4.57.0",
    "torch>=2.5.0",
    "torchaudio>=2.5.0",
    "numpy>=1.24.0"
)

# Require powerful GPUs (VoxCPM2 needs ~8GB VRAM)
image.require(gpu_name.in_([RTX_4090]))

# # Configure autoscaling
# app.configure_autoscaling(min_load=100)
app.configure_autoscaling(
    cold_workers=1,          # Idle workers to keep ready
    max_workers=2,          # Maximum concurrent workers
    min_load=2,            # Minimum load threshold to trigger scaling
    min_cold_load=1,        # Load threshold for cold workers
    target_util=0.9,         # Target utilization ratio (0-1)
    cold_mult=2,             # Cold worker multiplier
    inactivity_timeout=60,  # Seconds of inactivity before scaling down
)

# Ensure deployment is ready
app.ensure_ready()