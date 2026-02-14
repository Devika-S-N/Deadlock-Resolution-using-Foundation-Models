# VLM.py
# Vision-Language Model interface for local detour waypoint generation (LLaVA version)

import json
from typing import List, Tuple, Optional
from PIL import Image
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from llava.conversation import conv_templates


# ------------------ Setup ------------------

MODEL_PATH = "liuhaotian/llava-v1.5-7b"

print("Loading LLaVA model...")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH,
    None,
    model_name=get_model_name_from_path(MODEL_PATH),
    device="cuda"
)

model.eval()

print("LLaVA loaded on:", next(model.parameters()).device)


# ------------------ Prompt Builder ------------------

def _build_prompt(
    agent: Tuple[float, float],
    goal: Tuple[float, float],
    lidar_hits: List[Tuple[float, float]],
    yellow_waypoints: List[Tuple[float, float]],
    retry_note: Optional[str] = None,
) -> str:

    return f"""
<image>

You are assisting a mobile robot navigating in a 2D plane.

Agent position: {agent}
Goal position: {goal}
LIDAR hit points: {lidar_hits}
Previous detour waypoints: {yellow_waypoints}

Return exactly 4 LOCAL detour waypoints that:
- Avoid obstacles (stay 1.2 units away from red points)
- Avoid previous waypoints (stay 1.0 units away)
- Stay within 3 units of agent
- Stay within bounds [0, 15]

Respond with ONLY valid JSON.
Do NOT include explanation.
Each waypoint MUST be a list of exactly 2 numbers, where the first entry of each list is the x coordinate and the second entry of each list is the y coordinate.
Example:
{{
  "waypoints": [[3.2, 5.1], [4.0, 6.0], [5.0, 6.5], [6.0, 5.0]]
}}


""".strip()


# ------------------ Core API ------------------

def query_detour_waypoints(
    image_path: str,
    agent: Tuple[float, float],
    goal: Tuple[float, float],
    lidar_hits: List[Tuple[float, float]],
    yellow_waypoints: List[Tuple[float, float]] = None,
    retry_note: Optional[str] = None,
    logger=None,
) -> List[Tuple[float, float]]:

    if yellow_waypoints is None:
        yellow_waypoints = []

    prompt = _build_prompt(agent, goal, lidar_hits, yellow_waypoints, retry_note)

    if logger:
        logger.log("=" * 80)
        logger.log("LLaVA PROMPT:")
        logger.log(prompt)
        logger.log("=" * 80)

    # ----- Load Image -----
    image = Image.open(image_path).convert("RGB")

    image_tensor = process_images(
        [image],
        image_processor,
        model.config
    )[0].unsqueeze(0).to("cuda").half()

    # ----- Prepare conversation -----
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        return_tensors="pt"
    ).unsqueeze(0).to("cuda")

    # ----- Generate -----
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=300,
            temperature=0.,
            do_sample=False
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if logger:
        logger.log("LLaVA RAW OUTPUT:")
        logger.log(text)



    # ----- Parse JSON -----
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
        wps = data.get("waypoints", [])
        if any(len(wp) != 2 for wp in wps):
            raise RuntimeError("Invalid waypoint format from LLaVA")

        return [tuple(map(float, wp)) for wp in wps]
    except Exception as e:
        raise RuntimeError(f"Failed to parse LLaVA output: {e}")
