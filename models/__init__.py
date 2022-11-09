from .model import Model, Identity
from .sample_cnn import SampleCNN
from .conv_block import ConvBlock

########
# ISMIR models
########

from .tune_vanilla import TuneVanilla

# # extra contractive
from .contractive_experiments.tune_1_cont import Tune1Cont
from .contractive_experiments.tune_2_cont import Tune2Cont
from .contractive_experiments.tune_3_cont import Tune3Cont

# # extra expansive
from .expansive_experiments.tune_1_expa import Tune1Expa
from .expansive_experiments.tune_2_expa import Tune2Expa
from .expansive_experiments.tune_3_expa import Tune3Expa

# tail experiments
from .tail_experiments.tune_1_tail import Tune1Tail
from .tail_experiments.tune_2_tail import Tune2Tail
from .tail_experiments.tune_3_tail import Tune3Tail
from .tail_experiments.tune_4_tail import Tune4Tail
from .tune_5_tail import Tune5Tail

# tune with extra connections
from .tune_plus import TunePlus

# pretrained model experiment
from .additional_experiments.tune_hybrid_frozen import TuneHybridFrozen
from .additional_experiments.tune_hybrid_defrosted import TuneHybridDefrosted

#additional experiments (testing filter limiting performance)
from .additional_experiments.tune_plus_xl import TunePlusXL
from .additional_experiments.tune_5_tail_xl import Tune5TailXL
from .additional_experiments.tune_vanilla_xs import TuneVanillaXS

# evaluating different latent representation sizes with same parameter contraint (see section ~XX)
from .representation_size_experiments.tune_5_tail_64 import Tune5Tail64
from .representation_size_experiments.tune_5_tail_128 import Tune5Tail128
from .representation_size_experiments.tune_5_tail_256 import Tune5Tail256
from .representation_size_experiments.tune_5_tail_1024 import Tune5Tail1024







############
# no param constraint, thus all models with same initial number of filters
############

from .additional_experiments.tune_vanilla_free import TuneVanillaFree

# # extra contractive
from .contractive_experiments.tune_1_cont_free import Tune1ContFree
from .contractive_experiments.tune_2_cont_free import Tune2ContFree
from .contractive_experiments.tune_3_cont_free import Tune3ContFree

# # extra expansive
from .expansive_experiments.tune_1_expa_free import Tune1ExpaFree
from .expansive_experiments.tune_2_expa_free import Tune2ExpaFree
from .expansive_experiments.tune_3_expa_free import Tune3ExpaFree

# tail experiments
# from .tail_experiments.tune_1_tail_free import Tune1Tail # is the same as not free # filter-wise
from .tail_experiments.tune_2_tail_free import Tune2TailFree
from .tail_experiments.tune_3_tail_free import Tune3TailFree
from .tail_experiments.tune_4_tail_free import Tune4TailFree
from .tail_experiments.tune_5_tail_free import Tune5TailFree

# tune with extra connections between expansive and tail path
from .additional_experiments.tune_plus_free import TunePlusFree


