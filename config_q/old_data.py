from . import params

params['files_cfg'].update( 
    dict(
        obs_mask_path='/gpfsscratch/rech/nlu/commun/large/dataset_nadir_0d_swot.nc',
        obs_mask_var='ssh_mod',
        oi_path='/gpfsscratch/rech/nlu/commun/large/ssh_NATL60_swot_4nadir.nc',
        oi_var='ssh_mod',
    )
)
