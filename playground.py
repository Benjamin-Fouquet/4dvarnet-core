from new_dataloading import FourDVarNetDataModule
import torch
from hydra_main import FourDVarNetRunner
import xarray as xr
from torch.utils.data import Dataset
from solver import Solver_Grad_4DVarNN

'''file path definition in jz'''

path_torch = 'data/sample_batch_4dvarnet.torch'
path_xarray = 'data/Dataset_64_ZOI.npy'

fake_config = {
    'dim_range': ''

}


data = torch.load(path_torch)

dataloader = FourDVarNetDataModule(slice_win=5)

runner = FourDVarNetRunner(config=fake_config)



class OSSE_CSED(Dataset):

    mean = -3.6109390258789062
    std = 0.5700244903564453

    def __init__(
        self,
        data_path=Path(".") / "data/Dataset_64_ZOI_daily.nc",
        day_win=100,
        overlap=0.8,
        test=False,
        return_clouds=False,
    ):
        self.data_path = data_path
        self.day_win = day_win
        self.day_win_step = int(day_win * (1 - overlap))
        self.return_clouds = return_clouds

        self.ds = xr.load_dataset(data_path)
        self.mask = ~torch.tensor(self.ds["csed_daily"].isel(day=0).isnull().values).to(device=DEVICE)
        self.ds["csed_daily"] = (self.ds["csed_daily"] - self.mean) / self.std

        if test:
            self.ds = self.ds.isel(day=slice(-100, None))
        else:
            self.ds = self.ds.isel(day=slice(None, -100))

        ds = self.ds.fillna(0)
        self.csed = torch.tensor(ds["csed_daily"].values).to(device=DEVICE)
        self.cloud = torch.tensor(ds["cloud_daily"].values).to(device=DEVICE)

        def rolling_window(tsr, size, step):
            return tsr.unfold(0, size, step).moveaxis(-1, 1)

        self.csed_rolling = rolling_window(self.csed, self.day_win, self.day_win_step).to(device=DEVICE)
        self.cloud_rolling = rolling_window(self.cloud, self.day_win, self.day_win_step).to(device=DEVICE)

    def __len__(self):
        return len(self.csed_rolling)

    def __getitem__(self, idx):
        csed = self.csed_rolling[idx].unsqueeze(0)  # shape (C=1, D, H, W)
        cloud = self.cloud_rolling[idx].unsqueeze(0)  # shape (C=1, D, H, W)

        out = [csed * cloud, csed]

        if self.return_clouds:
            out.append(cloud)

        return tuple(o.float() for o in out)


#dim_range = {'lat': slice(25.0, 65.0, None), 'lon': slice(-73.0, 7.0, None), 'time': slice('2012-10-01', ...20', None)}