from goldmine.simulators.epidemiology import Epidemiology
from goldmine.simulators.epidemiology2d import Epidemiology2D
from goldmine.simulators.lotka_volterra import LotkaVolterra
from goldmine.simulators.galton import GeneralizedGaltonBoard
from goldmine.simulators.gaussian import GaussianSimulator
from goldmine.simulators.chutes_ladders import ChutesLaddersSimulator
from goldmine.inference.histograms import HistogramInference
from goldmine.inference.nde import MAFInference
from goldmine.inference.scandal import SCANDALInference
from goldmine.inference.rascandal import RASCANDALInference
from goldmine.inference.scandal_cv import SCANDALInferenceWithRatioControlVariate


def create_simulator(simulator_name):
    if simulator_name == 'gaussian':
        return GaussianSimulator()
    elif simulator_name == 'epidemiology':
        return Epidemiology()
    elif simulator_name == 'epidemiology2d':
        return Epidemiology2D()
    elif simulator_name == 'lotkavolterra':
        return LotkaVolterra()
    elif simulator_name == 'galton':
        return GeneralizedGaltonBoard()
    elif simulator_name == 'chutes_ladders':
        return ChutesLaddersSimulator()
    else:
        raise RuntimeError('Simulator name {} unknown'.format(simulator_name))


def create_inference(inference_name, **params):
    if inference_name == 'histogram':
        return HistogramInference(**params)
    elif inference_name == 'maf':
        return MAFInference(**params)
    elif inference_name == 'scandal':
        return SCANDALInference(**params)
    elif inference_name == 'rascandal':
        return RASCANDALInference(**params)
    elif inference_name == 'scandalcv':
        return SCANDALInferenceWithRatioControlVariate(**params)
    else:
        raise RuntimeError('Inference technique name {} unknown'.format(inference_name))
