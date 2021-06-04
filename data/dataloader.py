import pandas as pd
from torch.utils.data import Dataset
import operator
from torch.nn.functional import one_hot
from torch import tensor

class DatasetLoader(Dataset):
    def __init__(self, datapath, seq_length, max_apps):
        self.eventlog = pd.read_csv(datapath).Appname
        self.padlength = seq_length
        self.appindexdict = {}
        self.appnamedict = {}
        self.max_apps = max_apps

    def __len__(self):
        return len(self.eventlog)-self.padlength

    def __getitem__(self, idx):
        curr_app_name = self.eventlog.iloc[idx+self.padlength]
        curr_app_index = self.appindexdict.get(curr_app_name)
        if curr_app_index == None:
            curr_app_index = len(self.appindexdict)
            self.appindexdict[curr_app_name] = curr_app_index
            if idx == 0:
                for prev_app_name in self.eventlog.iloc[:idx+self.padlength]:
                    if (self.appindexdict.get(prev_app_name) == None):
                        self.appindexdict[prev_app_name] = len(self.appindexdict)
            self.appnamedict = dict(zip(self.appindexdict.values(), self.appindexdict.keys()))
        curr_app_index = tensor(curr_app_index)

        prev_app_names = self.eventlog.iloc[idx:idx+self.padlength]
        appindexgetter = operator.itemgetter(*prev_app_names)
        prev_apps_indices = tensor(appindexgetter(self.appindexdict))

        curr_app_onehot = one_hot(curr_app_index, self.max_apps)

        return prev_apps_indices, curr_app_index, curr_app_onehot

    def clean(self, removeapps: iter):
        self.eventlog.loc[self.eventlog == "à®µà®°à¯ˆà®ªà®Ÿà®®à¯"] = "Tamil Nadu Village Map"
        self.eventlog = self.eventlog[~self.eventlog.isin(removeapps)]
        self.eventlog = self.eventlog.reset_index(drop=True)