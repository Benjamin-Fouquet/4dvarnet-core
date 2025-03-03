from . import *
import yaml

params = yaml.safe_load("""
files_cfg:
  oi_path: /gpfsstore/rech/yrf/commun/NATL60/NATL/oi/ssh_NATL60_4nadir.nc
  oi_var: ssh_mod
  obs_target_path: /gpfswork/rech/yrf/commun/CalData/full_cal_obs.nc
  obs_target_var: swot
  obs_mask_path: /gpfswork/rech/yrf/commun/CalData/cal_data_karin_noise_only.nc
  obs_mask_var: nad_swot_roll_phase_bd_timing_karin
  gt_path: /gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc
  gt_var: ssh
splits:
  train_slices:
  - - '2012-10-01'
    - '2012-11-20'
  - - '2013-02-07'
    - '2013-09-30'
  test_slices:
  - - '2013-01-03'
    - '2013-01-27'
  val_slices:
  - - '2012-11-30'
    - '2012-12-24'
test_dates:
- '2013-01-03'
- '2013-01-04'
- '2013-01-05'
- '2013-01-06'
- '2013-01-07'
- '2013-01-08'
- '2013-01-09'
- '2013-01-10'
- '2013-01-11'
- '2013-01-12'
- '2013-01-13'
- '2013-01-14'
- '2013-01-15'
- '2013-01-16'
- '2013-01-17'
- '2013-01-18'
- '2013-01-19'
- '2013-01-20'
- '2013-01-21'
- '2013-01-22'
- '2013-01-23'
- '2013-01-24'
- '2013-01-25'
- '2013-01-26'
- '2013-01-27'
dataloading: new
data_dir: /gpfsscratch/rech/nlu/commun/large
dir_save: /gpfsscratch/rech/nlu/commun/large/results_maxime
iter_update:
- 0
- 20
- 40
- 60
- 100
- 150
- 800
nb_grad_update:
- 5
- 5
- 10
- 10
- 15
- 15
- 20
- 20
- 20
lr_update:
- 0.001
- 0.0001
- 0.001
- 0.0001
- 0.0001
- 1.0e-05
- 1.0e-05
- 1.0e-06
- 1.0e-07
k_batch: 1
n_grad: 5
dT: 5
dx: 1
W: 200
shape_state:
- 15
- 200
- 200
shape_obs:
- 10
- 200
- 200
dW: 3
dW2: 1
sS: 4
nbBlocks: 1
Nbpatches: 1
stochastic: false
animate: false
batch_size: 2
DimAE: 50
dim_grad_solver: 150
dropout: 0.25
dropout_phi_r: 0.0
alpha_proj: 0.5
alpha_sr: 0.5
alpha_lr: 0.5
alpha_mse_ssh: 10.0
alpha_mse_gssh: 1.0
sigNoise: 0.0
flagSWOTData: true
rnd1: 0
rnd2: 100
dwscale: 1
UsePriodicBoundary: false
InterpFlag: false
automatic_optimization: true
swot_anom_wrt: low_res
anom_swath_init: zeros
loss_glob: 1
loss_loc: 1
loss_proj: 1
loss_low_res: 1
w_loss:
- 0
- 0
- 1.0
- 0
- 0
mean_Tr: 0.3564856026938331
mean_Tt: 0.3564856026938331
mean_Val: 0.3564856026938331
var_Tr: 0.15653612895896984
var_Tt: 0.15653612895896984
var_Val: 0.15653612895896984
min_lon: -64.95
max_lon: -55.0
min_lat: 33.0
max_lat: 42.95
ds_size_time: 21.0
ds_size_lon: 1.0
ds_size_lat: 1.0
""")
