import os
import modal
import modal.gpu

app = modal.App(name="xtts")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
    )
    .pip_install_from_pyproject("pyproject.toml")
    .env({"COQUI_TOS_AGREED": "1"}) # Coqui requires you to agree to the terms of service before downloading the model
)

with image.imports():
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils.generic_utils import get_user_data_dir
    from TTS.utils.manage import ModelManager
    import torch
    # from TTS.api import TTS  

@app.cls(
    image=image,
    gpu="A10G",
    mounts=[modal.Mount.from_local_dir("server", remote_path="/root/server")]
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
    
    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, UploadFile, Body
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
        from typing import List
        import io
        import tempfile
        import torch
        import numpy as np
        import base64
        import wave

        from server.main import (
            postprocess,
            encode_audio_common,
            StreamingInputs,
            TTSInputs,
            predict_streaming_generator,
        )

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
