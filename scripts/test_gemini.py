# quick check that the api key works and gemini can read images
# usage: python scripts/test_gemini.py
#        python scripts/test_gemini.py --image path/to/image.jpg

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.llm.gemini import Gemini

parser = argparse.ArgumentParser()
parser.add_argument("--image")
args = parser.parse_args()

gemini = Gemini()
print(f"model: {gemini.model}")

if not args.image:
    print(gemini.ask("Say hi in one short sentence."))
else:
    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"could not read {args.image}")
    prompt = 'How many people are in this image and what are they doing? Answer as JSON: {"people": number, "description": "..."}'
    print(gemini.ask_with_images(prompt, [frame]))
