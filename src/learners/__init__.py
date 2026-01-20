from .q_learner import QLearner
from .nq_learner import NQLearner
from .dvd_learner import DVDNQLearner
# from .wm_fac_learner import WMFacLearner
from .dvd_wm_fac_learner import DVDWMFacLearner




#将不同的learner类注册到REGISTRY字典中
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["nq_learner"] = NQLearner
REGISTRY["dvd_wm_fac_learner"] = DVDWMFacLearner
REGISTRY["dvd_learner"] = DVDNQLearner


