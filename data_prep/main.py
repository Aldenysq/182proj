import os
from typing import List

import numpy as np
from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value
import skimage.io as skio
import json

DS_NAME = "aldenn13l/182-fine-tune"
DATA_ROOT = "data/images"
INSTRUCTIONS_PATH = 'data/instruction.json'
IMG_GAP = 50 # from img generation


def load_instructions(instructions_path: str) -> List[str]:
    f = open(instructions_path)
    data = json.load(f)
    return data


def generate_examples(image_paths: List[str], instructions: List[str]):
    def fn():
        for image_path in image_paths:
            # split image into two
            img = skio.imread(image_path)
            img_width = (img.shape[1] - IMG_GAP) // 2
            img1 = img[:, 0:img_width]
            img2 = img[:, img_width+IMG_GAP:]
            img_num = int(image_path.split('_')[1][:-4])
            path1 = 'data/temp/original_'+str(img_num)+'.png'
            skio.imsave(path1, img1)
            path2 = 'data/temp/new_'+str(img_num)+'.png'
            skio.imsave(path2, img2)
            yield {
                "original_image": {"path": path1},
                "edit_prompt": instructions[img_num],
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