"""Single entry point for asking the local VLM a question.
Deterministic settings come from config.py. There is NO retry: empty answers
are recorded as empty and counted as invalid in the analysis.
"""
import time

import ollama

import config

PREFIX = "This is a satellite image of an airport area. "


def ask(image_path, question):
    prompt = PREFIX + question
    messages = [{"role": "user", "content": prompt, "images": [str(image_path)]}]
    options = {
        "temperature": config.TEMPERATURE,
        "seed": config.SEED,
        "num_predict": config.MAX_NEW_TOKENS,
    }
    t0 = time.time()
    raw, err = "", ""
    try:
        try:
            resp = ollama.chat(model=config.MODEL, messages=messages,
                               options=options, think=config.THINK)
        except TypeError:
            # older ollama python package without the `think` argument
            resp = ollama.chat(model=config.MODEL, messages=messages,
                               options=options)
            err = "think-arg-unsupported"
        raw = (resp.message.content or "").strip()
    except Exception as e:  # keep the run going, record the problem
        err = str(e)
    return {"raw": raw, "seconds": round(time.time() - t0, 2), "error": err}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.vlm <image.png>")
        sys.exit()
    for q in ["Is there an aircraft in this image? Answer only Yes or No.",
              "How many aircraft are in this image? Answer with a single number."]:
        print(q, "->", ask(sys.argv[1], q))