
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

PARTS = [
    MODELS_DIR / "food_model.part1",
    MODELS_DIR / "food_model.part2",
]

OUTPUT = MODELS_DIR / "food_classifier_efficientnetb3_candidate.keras"

def assemble():
    missing = [part.name for part in PARTS if not part.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model parts: {missing}")

    with OUTPUT.open("wb") as destination:
        for part in PARTS:
            with part.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    destination.write(chunk)

    print(f"Model assembled: {OUTPUT}")
    print(f"Model size: {OUTPUT.stat().st_size:,} bytes")

if __name__ == "__main__":
    assemble()
