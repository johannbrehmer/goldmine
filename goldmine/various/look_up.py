from goldmine.simulators.epidemiology import Epidemiology
from goldmine.simulators.epidemiology2d import Epidemiology2D
from goldmine.simulators.lotka_volterra import LotkaVolterra
from goldmine.simulators.lotka_volterra_checkpointed import CheckpointedLotkaVolterra
from goldmine.simulators.galton import GeneralizedGaltonBoard
from goldmine.simulators.gaussian import GaussianSimulator
from goldmine.simulators.chutes_ladders import ChutesLaddersSimulator
from goldmine.inference.histograms import HistogramInference
from goldmine.inference.nde import MAFInference
from goldmine.inference.scandal import SCANDALInference
from goldmine.inference.rascandal import RASCANDALInference
from goldmine.inference.scandal_cv import SCANDALInferenceWithRatioControlVariate
from goldmine.inference.rolr import ROLRInference
from goldmine.inference.carl import CARLInference
from goldmine.inference.cascal import CASCALInference
from goldmine.inference.rascal import RASCALInference
from goldmine.inference.scandal_checkpointed import CheckpointedSCANDALInference


def create_simulator(simulator_name, checkpoint=False):
    if checkpoint:
        if simulator_name == 'lotkavolterra':
            return CheckpointedLotkaVolterra()
        else:
            raise RuntimeError('Checkpointed simulator name {} unknown'.format(simulator_name))
    else:
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


def create_inference(inference_name, checkpoint=False, **params):
    if checkpoint:
        if inference_name == 'scandal':
            return CheckpointedSCANDALInference(**params)
        else:
            raise RuntimeError('Checkpointed inference technique name {} unknown'.format(inference_name))

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
    elif inference_name == 'carl':
        return CARLInference(**params)
    elif inference_name == 'cascal':
        return CASCALInference(**params)
    elif inference_name == 'rolr':
        return ROLRInference(**params)
    elif inference_name == 'rascal':
        return RASCALInference(**params)
    else:
        raise RuntimeError('Inference technique name {} unknown'.format(inference_name))
