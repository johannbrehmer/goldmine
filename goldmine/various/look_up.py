from goldmine.simulators.epidemiology import Epidemiology
from goldmine.simulators.galton import GeneralizedGaltonBoard
from goldmine.inference.nde import MAFInference
from goldmine.inference.scandal import SCANDALInference


def create_simulator(simulator_name):
    if simulator_name == 'epidemiology':
        return Epidemiology()
    elif simulator_name == 'galton':
        return GeneralizedGaltonBoard()
    else:
        raise ValueError('Simulator name {} unknown'.format(simulator_name))


def create_inference(inference_name, **params):
    if inference_name == 'maf':
        return MAFInference(**params)
    elif inference_name == 'scandal':
        return SCANDALInference(**params)
    else:
        raise ValueError('Inference technique name {} unknown'.format(inference_name))
