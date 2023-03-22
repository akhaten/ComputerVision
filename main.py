from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from eagle.stitch import KitchenRosenfeld, Stitch, Beaudet

p = ArgumentParser()
p.add_argument("input", type=Path, nargs=2)
p.add_argument("output", type=Path)

args = p.parse_args()

CURDIR = Path(__file__).parent
IMG_DIR = CURDIR / Path("examples/images/tsukuba")

images = tuple(np.array(Image.open(filepath).convert("L").convert("F")) for filepath in args.input)

stitch = Stitch(
    images,
    detector=KitchenRosenfeld(),
    max_control_points=50,
)
output = stitch()
plt.imsave(args.output, output, cmap="gray")
