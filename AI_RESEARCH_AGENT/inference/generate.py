# from inference.model_loader import InferenceModel


# class TextGenerator:
#     def __init__(self, model: InferenceModel):
#         self.model = model

#     def run(self, instruction: str, context: str | None = None) -> str:
#         if context:
#             prompt = f"""### Instruction:
# {instruction}

# ### Context:
# {context}

# ### Response:
# """
#         else:
#             prompt = f"""### Instruction:
# {instruction}

# ### Response:
# """

#         return self.model.generate(prompt)
class TextGenerator:
    def __init__(self, inference_model):
        self.model = inference_model.model
        self.tokenizer = inference_model.tokenizer

    def run(self, instruction: str) -> str:
        prompt = f"""Below is an instruction. Write a response that completes the task.

### Instruction:
{instruction}

### Response:
"""

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            eos_token_id=self.tokenizer.encode("### Instruction:")[0],
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 🔥 HARD STOP: remove dataset continuation
        if "### Instruction:" in text:
            text = text.split("### Instruction:")[0]

        if "### Response:" in text:
            text = text.split("### Response:")[-1]

        return text.strip()
