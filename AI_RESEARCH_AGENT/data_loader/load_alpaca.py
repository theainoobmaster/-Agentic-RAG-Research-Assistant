# from datasets import load_dataset
# import json
# import os

# def load_and_convert_alpaca():
#     print("📥 Downloading Alpaca dataset from Hugging Face...")

#     # Load Alpaca dataset
#     dataset = load_dataset("tatsu-lab/alpaca")

#     # Ensure raw data directory exists
#     os.makedirs("data/raw", exist_ok=True)

#     converted_samples = []

#     for sample in dataset["train"]:
#         converted_samples.append({
#             "task": sample["instruction"],
#             "input": sample["input"] if sample["input"] else ""
#         })

#     output_path = "data/raw/tasks.json"

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(converted_samples, f, indent=2, ensure_ascii=False)

#     print(f"✅ Alpaca converted successfully")
#     print(f"📁 Saved {len(converted_samples)} samples to {output_path}")

# if __name__ == "__main__":
#     load_and_convert_alpaca()
from datasets import load_dataset
import json
import os

def load_and_convert_alpaca():
    print("📥 Downloading Alpaca dataset from Hugging Face...")

    dataset = load_dataset("tatsu-lab/alpaca")

    os.makedirs("data/raw", exist_ok=True)

    converted_samples = []
    MAX_SAMPLES = 10  # ✅ limit for practical training

    for i, sample in enumerate(dataset["train"]):
        if i >= MAX_SAMPLES:
            break

        converted_samples.append({
            "task": sample["instruction"],
            "input": sample["input"] if sample["input"] else ""
        })

    output_path = "data/processed/train.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_samples, f, indent=2, ensure_ascii=False)

    print("✅ Alpaca converted successfully")
    print(f"📁 Saved {len(converted_samples)} samples to {output_path}")

if __name__ == "__main__":
    load_and_convert_alpaca()
