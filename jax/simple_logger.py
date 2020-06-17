import numpy as np


class ObservationLists:
    def __init__(self):
        self.obs_dict = {}

    def obs(self, key, value):
        if key in list(self.obs_dict):
            self.obs_dict[key].append(value)
        else:
            self.obs_dict[key] = [value]

    def get(self, key):
        return self.obs_dict[key]


class PeriodAverager:
    def __init__(self):
        self.this_period_obs = ObservationLists()
        self.averaged_obs = ObservationLists()

    def obs(self, k, v):
        self.this_period_obs.obs(k, v)

    def end_period(self):
        for k in list(self.this_period_obs.obs_dict):
            self.averaged_obs.obs(k, np.mean(self.this_period_obs.get(k)))

    def get(self, key):
        return self.averaged_obs.get(key)


class Logger(object):
    # NOTE: usage
    # L = Logger()
    # for epoch in range(10):
    #     L.epoch.obs("asdf", epoch)
    #     L.epoch.obs("yutdf", epoch)
    #     for episode in range(20):
    #         L.epoch_avg.obs("hsdf", epoch ** episode)
    #         L.epoch_avg.obs("mnfgf", epoch - episode)
    #     L.end_epoch()

    # L.epoch.get("asdf")
    # L.epoch.get("yutdf")
    # L.epoch_avg.get("hsdf")
    # L.epoch.get("mnfgf")

    def __init__(self):
        self.epoch = ObservationLists()
        self.epoch_avg = PeriodAverager()
        self.episode = ObservationLists()
        self.episode_avg = ObservationLists()
        self.step = ObservationLists()

    def end_epoch(self):
        self.epoch_avg.end_period()
