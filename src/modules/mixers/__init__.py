from .qmix import QMixer

from .dvd import DVDMixer
from .nqmix import NQMixer
from .dvd_wm_fac_mixer import DVDWMFacMixer



REGISTRY = {
    "qmix": QMixer,
    "dvd": DVDMixer,
    "nqmix": NQMixer,
    "dvd_wm_fac": DVDWMFacMixer
}
