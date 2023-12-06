import os
from typing import List

import numpy as np
from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value
import json

DS_NAME = "aldenn13l/182-fine-tune"
DATA_ROOT = "data/original_images"
INSTRUCTIONS_PATH = 'data/instruction.json'
IMG_GAP = 50 # from img generation

new_image = "data/new_images/new_street_"


def load_instructions(instructions_path: str) -> List[str]:
    f = open(instructions_path)
    data = json.load(f)
    return data


def generate_examples(image_paths: List[str], instructions: List[str]):
    def fn():
        for image_path in image_paths:
            path1 = image_path
            img_num = path1[:-4].split('_')[3]
            path2 = new_image + img_num + '.png'
            yield {
                "original_image": {"path": path1},
                "edit_prompt": instructions[int(img_num)],
                "new_image": {"path": path2},
            }

    return fn


def main():
    instructions = load_instructions(INSTRUCTIONS_PATH)

    image_paths = os.listdir(DATA_ROOT)
    image_paths = [os.path.join(DATA_ROOT, d) for d in image_paths]

    generation_fn = generate_examples(image_paths, instructions)
    # print("Creating dataset...")
    ds = Dataset.from_generator(
        generation_fn,
        features=Features(
            original_image=ImageFeature(),
            edit_prompt=Value("string"),
            new_image=ImageFeature(),
        ),
    )

    print("Pushing to the Hub...")
    ds.push_to_hub(DS_NAME)


if __name__ == "__main__":
    main()