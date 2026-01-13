from inference.model_loader import InferenceModel
from inference.generate import TextGenerator
import torch
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "models/lora_adapter"

model = InferenceModel(
    base_model_id=BASE_MODEL,
    lora_path=LORA_PATH
     # use fp32 on CPU if needed
)
generator = TextGenerator(model)
prompt = f"""
Below is an instruction that describes a task.
Write a response that completes the task.

### Instruction:
Using ONLY the retrieved context below, answer the question.

If the answer is not explicitly present in the retrieved context,
respond exactly with:
"Not stated in the paper."

Retrieved Context:
The paper introduces a Transformer architecture based solely on self-attention,
removing recurrence and convolutions.

Question:
What datasets were used?

### Response:
"""

print(model.generate(prompt,max_new_tokens=20))


