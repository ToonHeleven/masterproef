import numpy as np
import math
from typing import Union

class EventConnectionGraph:

    """ CONSTRUCTOR FUNCTION THAT INITS A GRAPH AND CORRESPONDING DICT """
    def __init__(self, shape: (int, int), d: int, T: int):
        self.eventdict = {}
        self.appnamedict = {}
        self.__d = d
        self.__T = T
        self.graph = np.zeros(shape=shape, dtype=np.float32)

    """ UPDATES THE GRAPH GIVEN TWO APPNAMES AND TIME DIFFERENCE """
    def update(self, appnames, apptimes):
        currappname = appnames.iloc[-1]

        if (self.eventdict.get(currappname) is None):
            newappindex = len(self.eventdict)
            self.eventdict[currappname] = newappindex
            self.appnamedict[newappindex] = currappname

        currapptime = apptimes.iloc[-1]
        prevapptime = currapptime
        i = 2
        while (currapptime-prevapptime < self.__T) and (i < len(appnames)):
            currappindex = self.eventdict.get(currappname)
            prevappindex = self.eventdict.get(appnames.iloc[-i])
            prevapptime = apptimes.iloc[-i]
            self.graph[prevappindex, currappindex] += self.__calcdelta(timediff = currapptime-prevapptime)
            i += 1

    """ PREDICTS THE NEXT APP GIVEN A HISTORY OF AT LEAST ONE APP """
    def predict(self, apphistory: str, topk: int):
        if type(apphistory)==str:
            prevappindex = self.eventdict.get(apphistory)
            prevapprow = self.graph[prevappindex]
            return np.array([self.appnamedict.get(appindex) for appindex in (-prevapprow).argsort()[:topk]])
        else:
            return 1

    """ PRIVATE FUNCTION THAT CALCULATES THE WEIGHT ADDED TO GRAPH """
    def __calcdelta(self, timediff):

        delta = math.exp((-timediff)/self.__d)

        return delta