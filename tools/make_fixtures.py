"""Save three test-split leaves per class as small JPEGs for the unit tests."""

from pathlib import Path

import numpy as np
from PIL import Image

from edgeplant.config import ROOT
from edgeplant.data import load_split

OUT = ROOT / "tests" / "fixtures" / "leaves"
PER_CLASS = 3


def main() -> None:
    x, y = load_split("test")
    rng = np.random.default_rng(7)
    OUT.mkdir(parents=True, exist_ok=True)
    for cls in range(int(y.max()) + 1):
        for k, i in enumerate(rng.permutation(np.flatnonzero(y == cls))[:PER_CLASS]):
            Image.fromarray(np.asarray(x[i])).save(OUT / f"{cls:02d}_{k}.jpg", quality=92)
    print(len(list(Path(OUT).glob("*.jpg"))), "images")


if __name__ == "__main__":
    main()
