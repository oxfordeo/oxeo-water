class SelectBands:
    """Computes sample virtual arrays to numpy array"""

    def __init__(self, bands):
        self.bands = bands

    def __call__(self, sample):
        sample["data"] = sample["data"].sel({"bands": self.bands})
        return sample


class SelectConstellation:
    """Computes sample virtual arrays to numpy array"""

    def __init__(self, constellation):
        self.constellation = constellation

    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = sample[key][self.constellation]
        return sample


class Compute:
    """Computes sample virtual arrays to numpy array"""

    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = sample[key].values.astype("<i2")
        return sample
