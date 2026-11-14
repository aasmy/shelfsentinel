# quick checks for gemini
# usage: python scripts/test_gemini.py
#        python scripts/test_gemini.py --image path/to/image.jpg
#        python scripts/test_gemini.py --video path/to/video.mp4 --t 12.5

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.frames import get_frames
from shelfsentinel.llm.gemini import Gemini

parser = argparse.ArgumentParser()
parser.add_argument("--image")
parser.add_argument("--video")
parser.add_argument("--t", type=float, help="time in seconds to look at in the video")
args = parser.parse_args()

gemini = Gemini()
print(f"model: {gemini.model}")

if args.image:
    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"could not read {args.image}")
    prompt = 'How many people are in this image and what are they doing? Answer as JSON: {"people": number, "description": "..."}'
    print(gemini.ask_with_images(prompt, [frame]))

elif args.video:
    if args.t is None:
        sys.exit("give a time with --t")
    frames = get_frames(args.video, args.t)
    times = [ft for ft, _ in frames]
    print(f"sending frames at {times}s")
    prompt = (
        f"These are {len(frames)} frames from a store camera, in order, taken at {times} seconds. "
        "Describe only what the person's hands do with items (picking up, putting back, putting in pocket or bag). "
        'Answer as JSON: {"hands": "...", "item_taken": true/false, "item_hidden": true/false, "clear_view": true/false}'
    )
    print(gemini.ask_with_images(prompt, [f for _, f in frames]))

else:
    print(gemini.ask("Say hi in one short sentence."))
