from goldmine.simulators.epidemiology import Epidemiology
from goldmine.simulators.galton import GeneralizedGaltonBoard
from goldmine.inference.nde import MAFInference

def create_simulator(simulator_name):
    if simulator_name == 'epidemiology':
        return Epidemiology()
    elif simulator_name == 'galton':
        return GeneralizedGaltonBoard()
    else:
        raise ValueError('Simulator name {} unknown'.format(simulator_name))


def create_inference(inference_name):
    if inference_name == 'maf':
        return MAFInference
    else:
        raise ValueError('Inference technique name {} unknown'.format(inference_name))
