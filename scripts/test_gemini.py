# quick check that the api key works
# usage: python scripts/test_gemini.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.llm.gemini import Gemini

gemini = Gemini()
print(f"model: {gemini.model}")
print(gemini.ask("Say hi in one short sentence."))
