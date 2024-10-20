import io
import os
import tempfile
import modal
import numpy as np
import base64
import wave
from pydantic import BaseModel
from typing import List

app = modal.App(name="xtts")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
    )
    .pip_install_from_pyproject("pyproject.toml")
    .env({"COQUI_TOS_AGREED": "1"}) # Coqui requires you to agree to the terms of service before downloading the model
)

with image.imports():
    from trainer.io import get_user_data_dir
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils.manage import ModelManager
    import torch

class StreamingInputs(BaseModel):
    speaker_embedding: List[float]
    gpt_cond_latent: List[List[float]]
    text: str
    language: str
    add_wav_header: bool = True
    stream_chunk_size: str = "20"

class TTSInputs(BaseModel):
    speaker_embedding: List[float]
    gpt_cond_latent: List[List[float]]
    text: str
    language: str


def postprocess(wav):
    """Post process the output waveform"""
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav


def encode_audio_common(
    frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1
):
    """Return base64 encoded audio"""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    if encode_base64:
        b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
        return b64_encoded
    else:
        return wav_buf.read()



@app.cls(
    image=image,
    gpu="A100",
    # mounts=[modal.Mount.from_local_dir("server", remote_path="/root/server")],
    keep_warm=1,
)
class XTTS:
    @modal.build()
    @modal.enter()
    def load_model(self):
        # """
        # Load the model weights into GPU memory when the container starts.
        # """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading default model", flush=True)
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        print("Downloading XTTS Model:", model_name, flush=True)
        ModelManager().download_model(model_name)
        model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
        print("XTTS Model downloaded", flush=True)

        print("Loading XTTS", flush=True)
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=True if self.device == "cuda" else False)
        model.to(self.device)
        print("XTTS Loaded.", flush=True)
        self.model = model

    @modal.method()
    def nvidia_smi(self):
        import subprocess

        subprocess.run(["nvidia-smi"], check=True)

    @modal.method()
    def torch_cuda(self):
        import torch

        print(torch.cuda.get_device_properties("cuda:0"))
    
    @modal.method()
    def nvcc_version(self):
        import subprocess

        return subprocess.run(["nvcc", "--version"], check=True)

    @modal.method()
    def clone_speaker(self, audio: bytes):
        temp_audio_name = next(tempfile._get_candidate_names())
        with open(temp_audio_name, "wb") as temp, torch.inference_mode():
            temp.write(io.BytesIO(audio).getbuffer())
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                temp_audio_name
            )
        return {
            "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
            "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
        }
    
    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, UploadFile, Body
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
        from typing import List
        import io
        import tempfile
        import torch


        def predict_streaming_generator(parsed_input):
            speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
            gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
            text = parsed_input.text
            language = parsed_input.language

            stream_chunk_size = int(parsed_input.stream_chunk_size)
            add_wav_header = parsed_input.add_wav_header


            chunks = self.model.inference_stream(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                stream_chunk_size=stream_chunk_size,
                enable_text_splitting=True,
            )

            for i, chunk in enumerate(chunks):
                chunk = postprocess(chunk)
                if i == 0 and add_wav_header:
                    yield encode_audio_common(b"", encode_base64=False)
                    yield chunk.tobytes()
                else:
                    yield chunk.tobytes()

        web_app = FastAPI(
            title="XTTS Streaming server",
            description="""XTTS Streaming server""",
            version="0.0.1",
            docs_url="/",
        )

        @web_app.post("/clone_speaker")
        def predict_speaker(wav_file: UploadFile):
            temp_audio_name = next(tempfile._get_candidate_names())
            with open(temp_audio_name, "wb") as temp, torch.inference_mode():
                temp.write(io.BytesIO(wav_file.file.read()).getbuffer())
                gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                    temp_audio_name
                )
            return {
                "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
                "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
            }

        @web_app.post("/tts_stream")
        def predict_streaming_endpoint(parsed_input: StreamingInputs):
            return StreamingResponse(
                predict_streaming_generator(parsed_input),
                media_type="audio/wav",
            )

        @web_app.post("/tts")
        def predict_speech(parsed_input: TTSInputs):
            speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
            gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
            text = parsed_input.text
            language = parsed_input.language

            out = self.model.inference(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
            )

            wav = postprocess(torch.tensor(out["wav"]))

            return encode_audio_common(wav.tobytes())

        @web_app.get("/studio_speakers")
        def get_speakers():
            if hasattr(self.model, "speaker_manager") and hasattr(self.model.speaker_manager, "speakers"):
                return {
                    speaker: {
                        "speaker_embedding": self.model.speaker_manager.speakers[speaker]["speaker_embedding"].cpu().squeeze().half().tolist(),
                        "gpt_cond_latent": self.model.speaker_manager.speakers[speaker]["gpt_cond_latent"].cpu().squeeze().half().tolist(),
                    }
                    for speaker in self.model.speaker_manager.speakers.keys()
                }
            else:
                return {}

        @web_app.get("/languages")
        def get_languages():
            return self.model.config.languages

        return web_app

@app.local_entrypoint()
def main():
    xtts = XTTS()
    xtts.nvidia_smi.remote()
    xtts.torch_cuda.remote()
    xtts.nvcc_version.remote()
