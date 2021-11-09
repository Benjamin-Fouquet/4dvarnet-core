from .. import * 
from ..swot import params
import copy
params = copy.deepcopy(params)

params['model'] = 'vit'
params['swot_anom_wrt'] = 'low_res'
# params['swot_anom_wrt'] = 'high_res'
params['anom_swath_init'] = 'zeros'
# params['anom_swath_init'] = 'obs'
params['loss_glob'] = 1
params['loss_loc'] = 1
params['loss_proj'] = 0
params['loss_low_res'] = 0
#
params['alpha_loc_mse_ssh'] = 10.
params['alpha_loc_mse_gssh']  = 10.
params['drop_out_rate'] = 0.1
params['drop_out_attn'] = 0.1