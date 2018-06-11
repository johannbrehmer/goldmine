from goldmine.simulators.epidemiology import Epidemiology
from goldmine.simulators.galton import GeneralizedGaltonBoard

def create_simulator(simulator_name):
    if simulator_name == 'epidemiology':
        return Epidemiology()
    elif simulator_name == 'galton':
        return GeneralizedGaltonBoard()
    else:
        raise ValueError('Simulator name %s unknown'.format(simulator_name))


def create_inference(inference_name):
    raise ValueError('Inference technique name %s unknown'.format(inference_name))
