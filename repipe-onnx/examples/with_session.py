"""
Setup
    uv pip install repipe-onnx soundfile

Prepare models
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json

Run
    uv run python simple.py
"""

from piper_onnx import Piper
import soundfile as sf
import onnxruntime as ort


sess = ort.InferenceSession('en_US-ryan-medium.onnx', sess_options=ort.SessionOptions(), providers=['CPUExecutionProvider'])
piper = Piper.from_session(sess, config_path='en_US-ryan-medium.onnx.json')
samples, sample_rate = piper.create('Hello world from Piper!')
sf.write('audio.wav', samples, sample_rate)
print("Created audio.wav")