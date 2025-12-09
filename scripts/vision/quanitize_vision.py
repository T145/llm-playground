import requests
import torch
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot

#from llmcompressor.transformers.tracing import TraceableMllamaForConditionalGeneration
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

# Load model.
model_id = "T145/ZEUS-V22-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto"
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "flickr30k"
DATASET_SPLIT = {"calibration": "test[:512]"}
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Recipe
# recipe = [
#     GPTQModifier(
#         targets="Linear",
#         scheme="W4A16",
#         ignore=["re:.*lm_head", "re:multi_modal_projector.*", "re:vision_model.*"],
#     ),
# ]
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["re:.*lm_head", "re:multi_modal_projector.*", "re:vision_model.*"],
)

# Perform oneshot
# oneshot(
#     model=model,
#     tokenizer=model_id,
#     dataset=DATASET_ID,
#     splits=DATASET_SPLIT,
#     recipe=recipe,
#     max_seq_length=MAX_SEQUENCE_LENGTH,
#     num_calibration_samples=NUM_CALIBRATION_SAMPLES,
#     trust_remote_code_model=True,
#     data_collator=data_collator,
# )

oneshot(
    model=model,
    recipe=recipe,
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the animal in this image\n"},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
image_url = "https://www.thesprucepets.com/thmb/uQnGtOt9VQiML2oG2YzAmPErrHo=/5441x0/filters:no_upscale():strip_icc()/all-about-tabby-cats-552489-hero-a23a9118af8c477b914a0a1570d4f787.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw)

inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")

# Save to disk compressed.
SAVE_DIR = model_id.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
