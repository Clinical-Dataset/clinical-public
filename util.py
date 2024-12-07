import numpy

def calc_stdv(values:list[int]):
    std = numpy.std(values)
    mean = numpy.mean(values)

    return mean, std
