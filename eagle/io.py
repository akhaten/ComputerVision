import numpy
import matplotlib.pyplot
import pathlib
import enum
import scipy.io



def read(filename: pathlib.Path) -> numpy.ndarray:

    
    image: numpy.ndarray = None

    extension = filename.suffix
    
    if extension == 'npy':
        image = numpy.load(filename)
    # elif extension == '.mat':
    #     image = scipy.io.loadmat(filename)
    else:
        image = matplotlib.pyplot.imread(filename)

    return image


