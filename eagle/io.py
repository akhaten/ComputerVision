import numpy
import matplotlib.pyplot
import pathlib


def read(filename: pathlib.Path) -> numpy.ndarray|None:

    
    image: numpy.ndarray|None = None

    extension = filename.suffix
    
    if extension == 'npy':
        image = numpy.load(filename)
    # elif extension == '.mat':
    #     image = scipy.io.loadmat(filename)
    else:
        image = matplotlib.pyplot.imread(filename)

    return image


