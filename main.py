import sys
from pathlib import Path


# Allow `python main.py` without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agricultureai.evaluation.transformer import main  # noqa: E402


if __name__ == "__main__":
    main()