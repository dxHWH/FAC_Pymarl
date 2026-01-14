from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .ppo_learner import PPOLearner
from .lica_learner import LICALearner
from .nq_learner import NQLearner
from .policy_gradient_v2 import PGLearner_v2
from .max_q_learner import MAXQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .offpg_learner import OffPGLearner
from .fmac_learner import FMACLearner
from .cf_learner import DMAQ_qattenLearner as CF_Learner
from .nq_vae_learner import NQLearnerVAE
from .dvd_learner import DVDNQLearner

from .dvd_wm_learner import DVDWMLearner  # <--- 新增
from .dvd_rssm_learner import DVDRssmLearner
from .dvd_rssm_learner_origin import DVDRssmLearnerOrigin
from .dvd_wm_atten_learner import DVDWMAttenLearner
from .dvd_wm_causal_learner import DVDWMCausalLearner
from .wm_learner import WMLearner
# from .wm_fac_learner import WMFacLearner
from .dvd_wm_fac_learner import DVDWMFacLearner




#将不同的learner类注册到REGISTRY字典中
REGISTRY = {}

REGISTRY["q_learner"] = QLearner



