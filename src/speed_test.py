"""Times the local VLM on one image.
Run from the project root:
    python -m src.speed_test path/to/image.png
    python -m src.speed_test path/to/image.png qwen3-vl:2b
"""
import sys
import time

import ollama

import config

QUESTIONS = [
    "Is there a water area in this image? Answer only Yes or No.",
    "Is this a rural or an urban area? Answer with one word: rural or urban.",
    "How many buildings are visible? Answer with a single number.",
]


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.speed_test <image_path> [model]")
        return
    image_path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else config.MODEL
    print(f"Model: {model}")

    for q in QUESTIONS:
        start = time.time()
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": q, "images": [image_path]}],
            options={
                "temperature": config.TEMPERATURE,
                "seed": config.SEED,
                "num_predict": config.MAX_NEW_TOKENS,
            },
            think=config.THINK,
        )
        elapsed = time.time() - start
        answer = response.message.content.strip()
        print(f"{elapsed:6.1f}s | answer: {answer!r} | {q}")


if __name__ == "__main__":
    main()