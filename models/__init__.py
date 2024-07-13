from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
import yaml 

from importlib import import_module

def buildModel(netFile, cfgname):
    module = import_module('models.' + netFile)

    if netFile in ['AGIQA']:
        with open('./config/%s.yaml'%cfgname) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        model = getattr(module, 'CoOp')(cfg).model
        return model, cfg 
    else:
        raise RuntimeError('Invalid network name: {}.'.format(netFile))

    return model
