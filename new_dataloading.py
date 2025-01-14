print(f"Using current {__name__}")
import re
import numpy as np
import pytorch_lightning as pl
import xarray as xr
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import pandas as pd


def parse_resolution_to_float(frac):
    """ Matches a string consting of an integer followed by either a divisor
    ("/" and an integer) or some spaces and a simple fraction (two integers
    separated by "/").
    From https://stackoverflow.com/a/1806375
    Args:
        frac (str): resolution as string fraction or integer from config file
    Returns:
        float: resolution as float
    Example:
        for x in ['3', '1/12', '1/20', '1 2/3']: print(repr(parse_resolution_to_float(x)))
        >
        3.0
        0.0833333333333333
        0.05
        1.6666666666666665
    """
    frac_regex = re.compile(r'^(\d+)(?:(?:\s+(\d+))?/(\d+))?$')
    i, n, d = frac_regex.match(frac).groups()
    if d is None: n, d = 0, 1  # if d is None, then n is also None
    if n is None: i, n = 0, i
    return float(i) + float(n) / float(d)


def find_pad(sl, st, N):
    k = np.floor(N/st)
    if N>((k*st) + (sl-st)):
        pad = (k+1)*st + (sl-st) - N
    elif N<((k*st) + (sl-st)):
        pad = (k*st) + (sl-st) - N
    else:
        pad = 0
    return int(pad/2), int(pad-int(pad/2))

class XrDataset(Dataset):
    """
    torch Dataset based on an xarray file with on the fly slicing.
    """

    def __init__(
        self,
        path,
        var,
        slice_win,
        resolution=1/20,
        dim_range=None,
        strides=None,
        decode=False,
        resize_factor=1,
        compute=False
    ):
        """
        :param path: xarray file
        :param var: data variable to fetch
        :param slice_win: window size for each dimension {<dim>: <win_size>...}
        :param resolution: input spatial resolution 
        :param dim_range: Optional dimensions bounds for each dimension {<dim>: slice(<min>, <max>)...}
        :param strides: strides on each dim while scanning the dataset {<dim>: <dim_stride>...}
        :param decode: Whether to decode the time dim xarray (useful for gt dataset)
        :param compute: whether to convert dask arrays to xr.DataArray (caution memory)
        """
        super().__init__()
        self.var = var
        self.resolution = resolution
        # try/except block for handling both netcdf and zarr files
        try:
            _ds = xr.open_dataset(path, cache=False)
        except OSError as ex:
            _ds = xr.open_zarr(path)
        if decode:
            if str(_ds.time.dtype) == 'float64':
                _ds.time.attrs["units"] = "seconds since 2012-10-01"
                _ds = xr.decode_cf(_ds)
            else:
                _ds['time'] = pd.to_datetime(_ds.time)

        # rename latitute/longitude to lat/lon for consistency
        rename_coords = {}
        if not "lat" in _ds.coords and "latitude" in _ds.coords:
            rename_coords["latitude"] = "lat"
        if not "lon" in _ds.coords and "longitude" in _ds.coords:
            rename_coords["longitude"] = "lon"
        _ds = _ds.rename(rename_coords)

        # reshape
        if resize_factor!=1:
            _ds = _ds.coarsen(lon=resize_factor).mean(skipna=True).coarsen(lat=resize_factor).mean(skipna=True)
            self.resolution = self.resolution*resize_factor
        # dimensions
        self.ds = _ds.sel(**(dim_range or {}))
        self.Nt = self.ds.time.shape[0]
        self.Nx = self.ds.lon.shape[0]
        self.Ny = self.ds.lat.shape[0]
        # I) first padding x and y
        pad_x = find_pad(slice_win['lon'], strides['lon'], self.Nx)
        pad_y = find_pad(slice_win['lat'], strides['lat'], self.Ny)
        # get additional data for patch center based reconstruction
        dX = [pad_ *self.resolution for pad_ in pad_x]
        dY = [pad_ *self.resolution for pad_ in pad_y]
        dim_range_ = {
          'lon': slice(self.ds.lon.min().item()-dX[0], self.ds.lon.max().item()+dX[1]),
          'lat': slice(self.ds.lat.min().item()-dY[0], self.ds.lat.max().item()+dY[1]),
          'time': dim_range['time']
        }
        self.ds = _ds.sel(**(dim_range_ or {}))
        self.Nt = self.ds.time.shape[0]
        self.Nx = self.ds.lon.shape[0]
        self.Ny = self.ds.lat.shape[0]
        # II) second padding x and y
        pad_x = find_pad(slice_win['lon'], strides['lon'], self.Nx)
        pad_y = find_pad(slice_win['lat'], strides['lat'], self.Ny)
        # pad the dataset
        dX = [pad_ *self.resolution for pad_ in pad_x]
        dY = [pad_ *self.resolution for pad_ in pad_y]
        pad_ = {'lon':(pad_x[0],pad_x[1]),
                'lat':(pad_y[0],pad_y[1])}
        self.ds = self.ds.pad(pad_,
                              mode='reflect')
        self.Nx += np.sum(pad_x)
        self.Ny += np.sum(pad_y)
        # III) get lon-lat for the final reconstruction
        dX = ((slice_win['lon']-strides['lon'])/2)*self.resolution
        dY = ((slice_win['lat']-strides['lat'])/2)*self.resolution
        dim_range_ = {
          'lon': slice(dim_range_['lon'].start+dX, dim_range_['lon'].stop-dX),
          'lat': slice(dim_range_['lat'].start+dY, dim_range_['lat'].stop-dY),
        }
        self.lon = np.arange(dim_range_['lon'].start, dim_range_['lon'].stop, self.resolution)
        self.lat = np.arange(dim_range_['lat'].start, dim_range_['lat'].stop, self.resolution)
        self.slice_win = slice_win
        self.strides = strides or {}
        self.ds_size = {
            dim: max((self.ds.dims[dim] - slice_win[dim]) // self.strides.get(dim, 1) + 1, 0)
            for dim in slice_win
        }

        # reorder dimensions, this ensures dims ordering using
        # DataArray.data is consistent in numpy arrays (batch,time,lat,lon)
        self.ds = self.ds.transpose('time', 'lat', 'lon')

        # convert dask arrays to xr.DataArrays for faster computations
        if compute:
            self.ds = self.ds.compute()

    def __del__(self):
        self.ds.close()

    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
        

    def __getitem__(self, item):
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx,
                       self.strides.get(dim, 1) * idx + self.slice_win[dim])
            for dim, idx in zip(self.ds_size.keys(),
                                np.unravel_index(item, tuple(self.ds_size.values())))
        }
        return self.ds.isel(**sl)[self.var].data.astype(np.float32)


class FourDVarNetDataset(Dataset):
    """
    Dataset for the 4DVARNET method:
        an item contains a slice of OI, mask, and GT
        does the preprocessing for the item
    """

    def __init__(
        self,
        slice_win,
        dim_range=None,
        strides=None,
        oi_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/oi/ssh_NATL60_swot_4nadir.nc',
        oi_var='ssh_mod',
        oi_decode=False,
        obs_mask_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/data_new/dataset_nadir_0d_swot.nc',
        obs_mask_var='ssh_mod',
        obs_mask_decode=False,
        gt_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
        gt_var='ssh',
        gt_decode=True,
        sst_path=None,
        sst_var=None,
        sst_decode=True,
        resolution=1/20,
        resize_factor=1,
        compute=False
    ):
        super().__init__()

        self.oi_ds = XrDataset(
            oi_path, oi_var,
            slice_win=slice_win,
            resolution=resolution,
            dim_range=dim_range,
            strides=strides,
            decode=oi_decode,
            resize_factor=resize_factor,
            compute=compute
        )
        self.gt_ds = XrDataset(
            gt_path, gt_var,
            slice_win=slice_win,
            resolution=resolution,
            dim_range=dim_range,
            strides=strides,
            decode=gt_decode,
            resize_factor=resize_factor,
            compute=compute
        )
        self.obs_mask_ds = XrDataset(
            obs_mask_path, obs_mask_var,
            slice_win=slice_win,
            resolution=resolution,
            dim_range=dim_range,
            strides=strides,
            decode=obs_mask_decode,
            resize_factor=resize_factor,
            compute=compute
        )

        if sst_var is not None:
            self.sst_ds = XrDataset(
                sst_path, sst_var,
                slice_win=slice_win,
                resolution=resolution,
                dim_range=dim_range,
                strides=strides,
                decode=sst_decode,
                resize_factor=resize_factor,
                compute=compute
            )
        else:
            self.sst_ds = None

        self.norm_stats = (0, 1)
        self.norm_stats_sst = (0, 1)

    def set_norm_stats(self, stats, stats_sst=None):
        self.norm_stats = stats
        self.norm_stats_sst = stats_sst

    def __len__(self):
        return min(len(self.oi_ds), len(self.gt_ds), len(self.obs_mask_ds))

    def coordXY(self):
        # return self.gt_ds.lon, self.gt_ds.lat
        return self.gt_ds.ds.lon.data, self.gt_ds.ds.lat.data

    def __getitem__(self, item):
        mean, std = self.norm_stats
        _oi_item = self.oi_ds[item]
        _oi_item = (np.where(
            np.abs(_oi_item) < 10,
            _oi_item,
            np.nan,
        ) - mean) / std

        # glorys model has NaNs on land
        gt_item = (self.gt_ds[item] - mean) / std

        oi_item = np.where(~np.isnan(_oi_item), _oi_item, 0.)
        # obs_mask_item = self.obs_mask_ds[item].astype(bool) & ~np.isnan(oi_item) & ~np.isnan(_gt_item)

        _obs_item = (self.obs_mask_ds[item] - mean) / std
        obs_mask_item = ~np.isnan(_obs_item)
        obs_item = np.where(~np.isnan(_obs_item), _obs_item, np.zeros_like(_obs_item))

        if self.sst_ds == None:
            return oi_item, obs_mask_item, obs_item, gt_item
        else:
            mean, std = self.norm_stats_sst
            _sst_item = (self.sst_ds[item] - mean) / std
            sst_item = np.where(~np.isnan(_sst_item), _sst_item, 0.)

            return oi_item, obs_mask_item, obs_item, gt_item, sst_item

class FourDVarNetDataModule(pl.LightningDataModule):
    def __init__(
            self,
            slice_win,
            dim_range=None,
            strides=None,
            train_slices= (slice('2012-10-01', "2012-11-20"), slice('2013-02-07', "2013-09-30")),
            test_slices= (slice('2013-01-03', "2013-01-27"),),
            val_slices= (slice('2012-11-30', "2012-12-24"),),
            oi_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/oi/ssh_NATL60_swot_4nadir.nc',
            oi_var='ssh_mod',
            oi_decode=False,
            obs_mask_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/data/dataset_nadir_0d_swot.nc',
            obs_mask_var='ssh_mod',
            obs_mask_decode=False,
            gt_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
            gt_var='ssh',
            gt_decode=True,
            sst_path=None,
            sst_var=None,
            sst_decode=True,
            resize_factor=1,
            resolution="1/20",
            dl_kwargs=None,
            compute=False
    ):
        super().__init__()
        self.resize_factor = resize_factor
        self.dim_range = dim_range
        self.slice_win = slice_win
        self.strides = strides
        self.dl_kwargs = {
            **{'batch_size': 2, 'num_workers': 2, 'pin_memory': True},
            **(dl_kwargs or {})
        }
        self.oi_path = oi_path
        self.oi_var = oi_var
        self.oi_decode = oi_decode
        self.obs_mask_path = obs_mask_path
        self.obs_mask_var = obs_mask_var
        self.obs_mask_decode = obs_mask_decode
        self.gt_path = gt_path
        self.gt_var = gt_var
        self.gt_decode = gt_decode
        self.sst_path = sst_path
        self.sst_var = sst_var
        self.sst_decode = sst_decode

        self.resize_factor = resize_factor
        self.resolution  = parse_resolution_to_float(resolution)
        self.compute = compute

        self.train_slices, self.test_slices, self.val_slices = train_slices, test_slices, val_slices
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.norm_stats = None
        self.norm_stats_sst = None

    def compute_norm_stats(self, ds):
        sum = 0
        count = 0
        for gt in [_it for _ds in ds.datasets for _it in _ds.gt_ds]:
            sum += np.nansum(gt)
            count += np.sum(~np.isnan(gt))
        mean = sum / count
        sum = 0
        for gt in [_it for _ds in ds.datasets for _it in _ds.gt_ds]:
            sum += np.nansum((gt - mean)**2)
        std = (sum / count)**0.5

        if self.sst_var == None:
            return mean, std
        else:
            print('... Use SST data')
            mean_sst = float(xr.concat([_ds.sst_ds.ds[_ds.sst_ds.var] for _ds in ds.datasets], dim='time').mean())
            std_sst = float(xr.concat([_ds.sst_ds.ds[_ds.sst_ds.var] for _ds in ds.datasets], dim='time').std())

            return [mean, std], [mean_sst, std_sst]

    def set_norm_stats(self, ds, ns, ns_sst=None):
        for _ds in ds.datasets:
            _ds.set_norm_stats(ns, ns_sst)

    def get_domain_bounds(self, ds):
        min_lon = round(np.min(np.concatenate([_ds.gt_ds.ds['lon'].values for _ds in ds.datasets])), 2)
        max_lon = round(np.max(np.concatenate([_ds.gt_ds.ds['lon'].values for _ds in ds.datasets])), 2)
        min_lat = round(np.min(np.concatenate([_ds.gt_ds.ds['lat'].values for _ds in ds.datasets])), 2)
        max_lat = round(np.max(np.concatenate([_ds.gt_ds.ds['lat'].values for _ds in ds.datasets])), 2)
        return min_lon, max_lon, min_lat, max_lat

    def coordXY(self):
        return self.test_ds.datasets[0].coordXY()

    def get_domain_split(self):
        return self.test_ds.datasets[0].gt_ds.ds_size

    def setup(self, stage=None):
        self.train_ds, self.val_ds, self.test_ds = [
            ConcatDataset(
                [FourDVarNetDataset(
                    dim_range={**self.dim_range, **{'time': sl}},
                    strides=self.strides,
                    slice_win=self.slice_win,
                    oi_path=self.oi_path,
                    oi_var=self.oi_var,
                    oi_decode=self.oi_decode,
                    obs_mask_path=self.obs_mask_path,
                    obs_mask_var=self.obs_mask_var,
                    obs_mask_decode=self.obs_mask_decode,
                    gt_path=self.gt_path,
                    gt_var=self.gt_var,
                    gt_decode=self.gt_decode,
                    sst_path=self.sst_path,
                    sst_var=self.sst_var,
                    sst_decode=self.sst_decode,
                    resolution=self.resolution,
                    resize_factor=self.resize_factor,
                    compute=self.compute
                ) for sl in slices]
            )
            for slices in (self.train_slices, self.val_slices, self.test_slices)
        ]

        if self.sst_var == None:
            self.norm_stats = self.compute_norm_stats(self.train_ds)
            self.set_norm_stats(self.train_ds, self.norm_stats)
            self.set_norm_stats(self.val_ds, self.norm_stats)
            self.set_norm_stats(self.test_ds, self.norm_stats)
        else:
            self.norm_stats, self.norm_stats_sst = self.compute_norm_stats(self.train_ds)
            self.set_norm_stats(self.train_ds, self.norm_stats, self.norm_stats_sst)
            self.set_norm_stats(self.val_ds, self.norm_stats, self.norm_stats_sst)
            self.set_norm_stats(self.test_ds, self.norm_stats, self.norm_stats_sst)

        self.bounding_box = self.get_domain_bounds(self.train_ds)
        self.ds_size = self.get_domain_split()

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.dl_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.dl_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, **self.dl_kwargs, shuffle=False)


if __name__ == '__main__':
    """
    Test run for single batch loading and trainer.fit
    """

    # Specify the dataset spatial bounds
    dim_range = {
        'lat': slice(35, 45),
        'lon': slice(-65, -55),
    }

    # Specify the batch patch size
    slice_win = {
        'time': 5,
        'lat': 200,
        'lon': 200,
    }
    # Specify the stride between two patches
    strides = {
        'time': 1,
        'lat': 200,
        'lon': 200,
    }

    dm = FourDVarNetDataModule(
        slice_win=slice_win,
        dim_range=dim_range,
        strides=strides,
    )

    # Test a single batch loading
    dm.setup()
    dl = dm.val_dataloader()
    batch = next(iter(dl))
    oi, mask, gt = batch

    # Test fit
    from main import LitModel

    lit_mod = LitModel()
    trainer = pl.Trainer(gpus=1)
    # dm.setup()
    trainer.fit(lit_mod, datamodule=dm)
