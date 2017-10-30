
import warnings

def import_from(mdl, name):
    mdl = __import__(mdl, fromlist=[name])
    return getattr(mdl, name)


#required, default (if not required), type, subtype*(if previous type is list or tuple)
parameter_specs = {
    'cpu'                      :[True, None, bool],
    'epochs'                   :[True, None, str],
    'network_type'             :[True, None, str],
    'finetune'                 :[True, None, bool],
    'out_wght_path'            :[True, None, str],
    'decay'                    :[True, None, float],
    'lr'                       :[True, None, float],
    'decay_at_epoch'           :[True, None, list, int],
    'factor_at_epoch'          :[True, None, list, float],
    'progress_logging'         :[True, None, bool],
    'batch_size'               :[True, None, int],
    'kernel_lr_multiplier'     :[True, None, float],
    'tensorboard_name'         :[True, None, str],
    'kernel_regularizer'       :[True, None, float],
    'activity_regularizer'     :[True, None, float],
    'bits'                    : [False, None, int],
    'wbits'                    :[False, None, int],
    'abits'                    :[False, None, int],
    'nla'                      :[False, None, int],
    'nfa'                      :[False, None, int],
    'nlb'                      :[False, None, int],
    'nfb'                      :[False, None, int],
    'nlc'                      :[False, None, int],
    'nfc'                      :[False, None, int],
    'dataset'                      :[False, None, int],
    'dim'                      :[False, None, int],
    'channels'                      :[False, None, int],
    'classes'                      :[False, None, int],
    }

def parse_param(param, value):
    #todo: support complex types ( (nested) lists/tuples...)
    if isinstance(value, parameter_specs[param][2]):
        return value
    elif not parameter_specs[param][0]: # if not required, check if None
        if value in ['None', 'none', '']:
            return None

    return parameter_specs[param][2](value)

class Config:
    def __init__(self, cfg, cmd_args = {}):
        try:
            
            for k in parameter_specs:
                self.proces_param(k, cfg, cmd_args)

        except ImportError:
            print('The configfile you provided ({}) cannot be imported, please verify.'.format(cfg))
            exit(1)


        self.postprocess()
    
    def proces_param(self, param, cfg, cmd_args):
        if param in cmd_args :
            setattr(self, param.lower(), parse_param(param, cmd_args[param]))
        elif param.lower() in cmd_args:
            setattr(self, param.lower(), parse_param(param, cmd_args[param.lower()]))
        else:
            try:
                setattr(self, param.lower(),import_from('config.{}'.format(cfg), param))
            except AttributeError:
                if parameter_specs[param][0]: #if required
                    raise 
                else:
                    setattr(self, param.lower(), parameter_specs[param][1])


    def postprocess(self):
        if hasattr(self, 'bits') and self.bits is not None:
            if self.abits is None:
                self.abits=self.bits
                warnings.warn('specialized bits to abits')
            if self.wbits is None:
                self.wbits = self.bits
                warnings.warn('specialized bits to wbits')
        del self.bits #to make sure it is not further used
        if hasattr(self, 'class'):
            self.clss=getattr(self,'class')
        self.out_wght_path = './weights/{}_{}_{}b_{}b_{}_{}_{}_{}_{}_{}.hdf5'.format(self.dataset,self.network_type, self.abits, self.wbits, self.nla, self.nfa, self.nlb,
                                                                             self.nfb, self.nlc, self.nfc)
        self.tensorboard_name = '{}_{}_{}b_{}b_{}_{}_{}_{}_{}_{}.hdf5'.format(self.dataset,self.network_type, self.abits, self.wbits, self.nla, self.nfa, self.nlb,
                                                                             self.nfb, self.nlc, self.nfc)