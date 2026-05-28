import asyncio
import base64
import io
from deployOne import app, generate_speech

async def main():
    """
    Test the VoxCPM2 TTS deployment with various modes:
    1. Basic text-to-speech
    2. Voice design (with description)
    3. Voice cloning (if reference audio available)
    """
    
    print("=" * 60)
    print("VoxCPM2 Text-to-Speech Deployment Client")
    print("=" * 60)
    
    # Test 1: Basic Text-to-Speech
    print("\n[Test 1] Basic Text-to-Speech Generation...")
    with open("./samplevoice.wav", "rb") as f:
        reference_wav_hex = f.read().hex()

    result = await generate_speech(
        text="(A gentle man voice, feeling happy when talking) ខ្ញុំមានឈ្មោះលីហ៊ូសំបូស្នេហ៍ ហាសហាស, មិនចាញបង ហ៊ុន នៅក្រវ៉ាញឡើយ !!! ខ្ញុំនឹងញ៉ែលីសាអោយបាន ",
        reference_wav_hex=reference_wav_hex,
    )
    
    if result["status"] == "success":
        print(f"✓ Audio generated successfully")
        print(f"  Sample Rate: {result['sample_rate']} Hz")
        print(f"  Duration: {result['duration_seconds']:.2f} seconds")
        
        # Decode and save audio
        audio_data = bytes.fromhex(result['audio'])
        with open("output_basic.wav", "wb") as f:
            f.write(audio_data)
        print(f"  Saved to: output_basic.wav")
    else:
        print(f"✗ Error: {result['error']}")
    
    # Test 2: Voice Design (Creative Voice)
    print("\n[Test 2] Voice Design - Creative Voice Generation...")
    result = await generate_speech(
        text="(A young woman, gentle and sweet voice)  ខ្ញុំមានឈ្មោះលីសា ស្រលាញបងហ៊ូ គាត់សំបូស្នេហ៍ គាត់ពូកែញ៉ែណាស !! ",
        cfg_value=2.0
    )
    
    
    if result["status"] == "success":
        print(f"✓ Creative voice generated successfully")
        print(f"  Sample Rate: {result['sample_rate']} Hz")
        print(f"  Duration: {result['duration_seconds']:.2f} seconds")
        
        audio_data = bytes.fromhex(result['audio'])
        with open("output_voice_design.wav", "wb") as f:
            f.write(audio_data)
        print(f"  Saved to: output_voice_design.wav")
    else:
        print(f"✗ Error: {result['error']}")
    
    # Test 3: Multiple requests for robustness
    print("\n[Test 3] Batch Generation...")
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "VoxCPM2 supports 30 different languages.",
        "This is a streaming TTS demonstration."
    ]
    
    for i, prompt in enumerate(prompts, 1):
        result = await generate_speech(text=prompt)
        if result["status"] == "success":
            print(f"✓ Batch {i}/3: Generated {result['duration_seconds']:.2f}s audio")
        else:
            print(f"✗ Batch {i}/3: Failed - {result['error']}")
    
    print("\n" + "=" * 60)
    print("Deployment test completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())