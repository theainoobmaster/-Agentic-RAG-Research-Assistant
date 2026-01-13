from inference.model_loader import InferenceModel


BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "models/lora_adapter"


def test_model_and_tokenizer_load():
    """
    Sanity check: base model + tokenizer + LoRA load correctly.
    """
    model = InferenceModel(
        base_model_id=BASE_MODEL,
        lora_path=LORA_PATH,
    )

    assert model is not None
    assert model.model is not None
    assert model.tokenizer is not None


def test_generate_returns_text():
    """
    Sanity check: generate() returns non-empty text.
    """
    model = InferenceModel(
        base_model_id=BASE_MODEL,
        lora_path=LORA_PATH,
    )

    prompt = "Explain neural networks in one sentence."

    output = model.generate(
        prompt,
        max_new_tokens=50,
    )

    assert isinstance(output, str)
    assert len(output.strip()) > 0


def test_generation_is_reasonable_language():
    """
    Guardrail: output should look like language, not junk.
    """
    model = InferenceModel(
        base_model_id=BASE_MODEL,
        lora_path=LORA_PATH,
    )

    prompt = "What is machine learning?"

    output = model.generate(prompt, max_new_tokens=50)

    # Very light checks (do NOT over-test semantics)
    assert len(output) > 15
    assert "\n" in output or " " in output

if __name__ == "__main__":
    print("Running inference sanity tests....")
    test_model_and_tokenizer_load()
    test_generate_returns_text()
    test_generation_is_reasonable_language()
    print("All tests passed!")