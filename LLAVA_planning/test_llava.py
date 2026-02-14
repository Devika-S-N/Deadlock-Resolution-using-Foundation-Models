import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    None,
    model_name=get_model_name_from_path(model_path),
    device="cuda"
)

print(next(model.parameters()).device)

# Load image
image_path = "Trial_image/vlm_snapshot_18:37:33_12-02_027.png"
image = Image.open(image_path).convert("RGB")

image_tensor = process_images(
    [image],
    image_processor,
    model.config
)[0].unsqueeze(0).to("cuda").half()


# Prompt
prompt = "<image>\nDescribe what you see in this image."


conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    return_tensors="pt"
).unsqueeze(0).to("cuda")

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.2
    )

outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(outputs)
