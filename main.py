from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from eagle.stitch import KitchenRosenfeld, Stitch, Beaudet

# p = ArgumentParser()
# p.add_argument("input", type=Path, nargs="+")

# args = p.parse_args()

input = [Path("examples/images") / f"pano{i+1}.jpeg" for i in range(3)]

stitch = Stitch(
    [np.array(Image.open(inp).convert("L").convert("F")) for inp in input],
    detector=KitchenRosenfeld(),
    max_control_points=8,
)
stitch()
