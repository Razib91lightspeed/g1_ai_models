#!/usr/bin/env python3

"""
Qwen3-VL inference API for the G1 Physical AI experiment.

Runs on NVIDIA DGX Spark.
DEVBOX communicates through an SSH tunnel.
This server performs inference only and cannot control the robot.
"""

import base64
import io
import os
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_PATH = os.environ["MODEL_PATH"]

app = FastAPI(
    title="G1 Spark VLM API",
    version="0.1.0"
)

inference_lock = Lock()


# --------------------------------------------------
# 2. Load the model
# --------------------------------------------------

print("Loading Qwen3-VL model...", flush=True)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
    local_files_only=True
)

model.eval()

print("Qwen3-VL model ready.", flush=True)


# --------------------------------------------------
# 3. Request format
# --------------------------------------------------

class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int = Field(default=128, ge=1, le=256)
    temperature: float = 0.0


# --------------------------------------------------
# 4. List available models
# --------------------------------------------------

@app.get("/v1/models")
def list_models():

    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model"
            }
        ]
    }


# --------------------------------------------------
# 5. Health check
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": MODEL_ID,
        "cuda_available": torch.cuda.is_available()
    }


# --------------------------------------------------
# 6. Image and text inference
# --------------------------------------------------

@app.post("/v1/chat/completions")
def chat(request: ChatRequest):

    if request.model != MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail="Unknown model"
        )

    if request.temperature != 0:
        raise HTTPException(
            status_code=400,
            detail="Only deterministic decoding is supported"
        )

    if len(request.messages) != 1:
        raise HTTPException(
            status_code=400,
            detail="Expected exactly one user message"
        )

    message = request.messages[0]

    if message.get("role") != "user":
        raise HTTPException(
            status_code=400,
            detail="Expected a user message"
        )

    content = message.get("content")

    if not isinstance(content, list):
        raise HTTPException(
            status_code=400,
            detail="Expected multimodal content"
        )

    processed_content = []
    image_count = 0
    text_count = 0

    for item in content:

        if not isinstance(item, dict):
            raise HTTPException(400, "Invalid content item")

        # Text input
        if item.get("type") == "text":

            value = item.get("text", "")

            if not isinstance(value, str) or len(value) > 2000:
                raise HTTPException(400, "Invalid text")

            text_count += 1

            processed_content.append({
                "type": "text",
                "text": value
            })

        # Image input
        elif item.get("type") == "image_url":

            image_count += 1

            image_url = item.get("image_url", {})

            if not isinstance(image_url, dict):
                raise HTTPException(400, "Invalid image URL")

            url = image_url.get("url", "")

            prefix = "data:image/jpeg;base64,"

            if not isinstance(url, str) or not url.startswith(prefix):
                raise HTTPException(
                    status_code=400,
                    detail="JPEG base64 image required"
                )

            try:
                # Decode the image
                encoded = url[len(prefix):]
                raw = base64.b64decode(
                    encoded,
                    validate=True
                )

                if len(raw) > 5_000_000:
                    raise ValueError("Image exceeds 5 MB")

                # Resize to limit inference memory use
                with Image.open(io.BytesIO(raw)) as source:
                    source.thumbnail((768, 768))
                    image = source.convert("RGB")

            except (
                ValueError,
                UnidentifiedImageError,
                OSError
            ) as exc:

                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image: {exc}"
                )

            processed_content.append({
                "type": "image",
                "image": image
            })

        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported content type"
            )

    if text_count != 1 or image_count != 1:
        raise HTTPException(
            status_code=400,
            detail="Exactly one text and one image required"
        )

    # --------------------------------------------------
    # 7. Prepare model input
    # --------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": processed_content
        }
    ]

    # --------------------------------------------------
    # 8. Generate response
    # --------------------------------------------------

    with inference_lock, torch.inference_mode():

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=False
        )

        input_length = inputs["input_ids"].shape[-1]

        generated_tokens = outputs[:, input_length:]

        answer = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]

    # --------------------------------------------------
    # 9. Return OpenAI-compatible response
    # --------------------------------------------------

    return {
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ]
    }

