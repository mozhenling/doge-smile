from .Human import Human
from .BaseChain import BaseChain
from .InContext import InContext
from .InContextHIs import InContextHIs
from .InContextRaw import InContextRaw

from .Smile import Smile
from .SmileAgent import SmartAgent
from .SmileAblation import SmileAblation
from .AFTD import AFTD


from .ERM import ERM
from .ERMpp import ERMpp

from .MixStyle import MixStyle
from .URM import URM
from .RIDG import RIDG
from .VNE import VNE

from .AGLU import AGLU
from .RDR import RDR
from .iBRF import iBRF
from .ELMloss import ELMloss

# If you prefer explicit mapping (more control, less magic), you can create a registry:

# ALGORITHMS = {
#     "AlgorithmA": AlgorithmA,
#     "AlgorithmB": AlgorithmB,
#     "AlgorithmC": AlgorithmC,
# }