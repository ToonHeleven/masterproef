import pandas as pd
from torch.utils.data import Dataset
import operator
from torch.nn.functional import one_hot
from torch import tensor
import numpy as np
import datetime
import math


class DatasetLoader(Dataset):
    def __init__(self, datapath, seq_length):
        self.eventlog = pd.read_csv(datapath).Appname
        self.padlength = seq_length
        self.appindexdict = {}
        self.appnamedict = {}

    def __len__(self):
        return len(self.eventlog)-self.padlength

    def __getitem__(self, idx):
        curr_app_name = self.eventlog.iloc[idx+self.padlength]
        curr_app_index = self.appindexdict.get(curr_app_name)
        if curr_app_index == None:
            if idx == 0:
                for prev_app_name in self.eventlog.iloc[:idx+self.padlength]:
                    if (self.appindexdict.get(prev_app_name) == None):
                        self.appindexdict[prev_app_name] = len(self.appindexdict)
            curr_app_index = len(self.appindexdict)
            self.appindexdict[curr_app_name] = curr_app_index
            self.appnamedict = dict(zip(self.appindexdict.values(), self.appindexdict.keys()))
        curr_app_index = tensor(curr_app_index)

        prev_app_names = self.eventlog.iloc[idx:idx+self.padlength]
        appindexgetter = operator.itemgetter(*prev_app_names)
        prev_apps_indices = tensor(appindexgetter(self.appindexdict))

        return prev_apps_indices, curr_app_index

    def clean(self, removeapps: iter):
        self.eventlog.loc[self.eventlog == "à®µà®°à¯ˆà®ªà®Ÿà®®à¯"] = "Tamil Nadu Village Map"
        self.eventlog = self.eventlog[~self.eventlog.isin(removeapps)]
        self.eventlog = self.eventlog.reset_index(drop=True)


""" Event Connection Graph wordt als DatasetLoader opgesteld """
class EventConnectionGraph(Dataset):

    """ Bij initialisatie wordt de event connection graph aangemaakt """
    def __init__(self, datapath: str, d: int, T: int, removeapps: list, nr_generated: int, nr_samples: int = None):
        if nr_samples is None:
            raise ValueError("Define nr_samples")

        self.eventlog = pd.read_csv(datapath)[["Appname", "Timestamp"]]
        self.clean(removeapps)
        if (nr_samples is not None):
            if nr_samples>len(self.eventlog): raise ValueError("nr_samples is larger than dataset")
            self.eventlog = self.eventlog.iloc[:nr_samples]

        self.eventlog["Seconds"] = self.eventlog.Timestamp.apply(lambda x: (datetime.datetime.strptime(x, "%d/%m/%Y %H:%M:%S") - datetime.datetime(1970, 1, 1)).total_seconds())
        self.nr_apps = len(self.eventlog.Appname.drop_duplicates())
        self.__d = d
        self.__T = T
        self.appindexdict = {}
        self.appnamedict = {}
        self.graph = np.zeros(shape=(self.nr_apps, self.nr_apps), dtype=np.float64)
        self.nr_generated = nr_generated

        for curr_app_pos in range(len(self.eventlog)):
            curr_app_name = self.eventlog.Appname.iloc[curr_app_pos]
            curr_app_index = self.appindexdict.get(curr_app_name)

            if (curr_app_index is None):
                curr_app_index = len(self.appindexdict)
                self.appindexdict[curr_app_name] = curr_app_index
                self.appnamedict[curr_app_index] = curr_app_name

            prev_app_pos = curr_app_pos-1
            while prev_app_pos > 0 and (self.eventlog.Seconds[curr_app_pos]-self.eventlog.Seconds[prev_app_pos]) < self.__T:
                prev_app_name = self.eventlog.Appname[prev_app_pos]
                prev_app_index = self.appindexdict.get(prev_app_name)
                self.graph[prev_app_index, curr_app_index] = self.__calcdelta(self.eventlog.Seconds[curr_app_pos]-self.eventlog.Seconds[prev_app_pos])
                prev_app_pos -= 1

        self.graph = np.divide(self.graph.T, self.graph.sum(axis=1),
                               where=self.graph.sum(axis=1) != 0,
                               out=np.full(shape=(self.nr_apps, self.nr_apps), fill_value=1/self.nr_apps)).T

    """ Wat normaal de lengte is van de dataset in DatasetLoader is nu het gewenst aan gegenereerde samples """
    def __len__(self):
        return self.nr_generated

    """ Specifieert wat de DataLoader als samples produceert, in dit geval een willekeurige app index en een opvolger obv. de graph"""
    def __getitem__(self, idx):
        prev_app_index = np.random.randint(0, self.nr_apps - 1)
        next_app_index = np.random.choice(self.nr_apps, p=self.graph[prev_app_index])
        return prev_app_index, next_app_index

    """ Functie die de event log cleant """
    def clean(self, removeapps: iter):
        self.eventlog.loc[self.eventlog.Appname == "à®µà®°à¯ˆà®ªà®Ÿà®®à¯"] = "Tamil Nadu Village Map"
        self.eventlog = self.eventlog[~self.eventlog.Appname.isin(removeapps)]
        self.eventlog = self.eventlog.reset_index(drop=True)

    """ private functie op de delta te berekenen in aanmaken van event connection graph """
    def __calcdelta(self, timediff):
        return math.exp((-timediff) / self.__d)