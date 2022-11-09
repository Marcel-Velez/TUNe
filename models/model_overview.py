from torchsummary import summary
import torch
import argparse 

from . import TuneVanilla

from . import Tune1Cont, Tune2Cont, Tune3Cont
from . import Tune1Expa, Tune2Expa, Tune3Expa
from . import Tune1Tail, Tune2Tail, Tune3Tail, Tune4Tail, Tune5Tail

from . import TunePlus

from . import TuneHybridDefrosted, TuneHybridFrozen
from . import Tune5TailXL, TunePlusXL, TuneVanillaXS 
from . import Tune5Tail64, Tune5Tail128, Tune5Tail256, Tune5Tail1024

# free experiments
from . import TuneVanillaFree

from . import Tune1ContFree, Tune2ContFree, Tune3ContFree
from . import Tune1ExpaFree, Tune2ExpaFree, Tune3ExpaFree
from . import Tune2TailFree, Tune3TailFree, Tune4TailFree, Tune5TailFree

from . import TunePlusFree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model_overview")
    # parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--eval",type=int,default=None)
    args = parser.parse_args()
    

    input_shape = (1,(59049))

    # # # # TuneVanilla
    print("New in: TuneVanilla")
    model = TuneVanilla(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)



    print("\n\n ", "-"*20, "\nContractive experiments\n", "-"*20,"\n\n")
    # # # # Tune1Cont
    print("New in: Tune1Cont")
    model = Tune1Cont(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # Tune2Cont
    print("New in: Tune2Cont")
    model = Tune2Cont(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # Tune3Cont
    print("New in: Tune3Cont")
    model = Tune3Cont(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)

    print("\n\n ", "-"*20, "\Expansive experiments\n", "-"*20,"\n\n")
    # # # # # Tune1Expa
    print("New in: Tune1Expa")
    model = Tune1Expa(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # Tune2Expa
    print("New in: Tune2Expa")
    model = Tune2Expa(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # # Tune3Expa
    print("New in: Tune3Expa")
    model = Tune3Expa(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)

    print("\n\n ", "-"*20, "\Tail experiments\n", "-"*20,"\n\n")
     # # # # Tune1Tail
    print("New in: Tune1Tail")
    model = Tune1Tail(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # Tune2Tail
    print("New in: Tune2Tail")
    model = Tune2Tail(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # Tune3Tail
    print("New in: Tune3Tail")
    model = Tune3Tail(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # Tune4Tail
    print("New in: Tune4Tail")
    model = Tune4Tail(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # Tune5Tail
    print("New in: Tune5Tail")
    model = Tune5Tail(n_classes=50, eval_layer=args.eval)
    summary(model.to(torch.device('cuda')), input_shape)


    #### must have the model chekpoints in the correct directory
    # print("\n\n ", "-"*20, "\Pretrained experiments\n", "-"*20,"\n\n")
    # # # # TuneHybridDefrosted
    # print("New in: TuneHybridDefrosted")
    # model = TuneHybridDefrosted(n_classes=50)
    # summary(model.to(torch.device('cuda')), input_shape)

    # # # # # # TuneHybridFrozen
    # print("New in: TuneHybridFroze")
    # model = TuneHybridFrozen(n_classes=50)
    # summary(model.to(torch.device('cuda')), input_shape)


    # # # # # TunePlus
    print("New in: TunePlus")
    model = TunePlus(n_classes=50, eval_layer=int(args.eval))
    summary(model.to(torch.device('cuda')), input_shape)

    # # # # Tune5TailXL
    print("New in: Tune5TailXL")
    model = Tune5TailXL(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # TunePlusXL
    print("New in: TunePlusXL")
    model = TunePlusXL(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # # TuneVanillaXS
    print("New in: TuneVanillaXS")
    model = TuneVanillaXS(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)

    # # # Tune5Tail64
    print("New in: Tune5Tail64")
    model = Tune5Tail64(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # Tune5Tail128
    print("New in: Tune5Tail128")
    model = Tune5Tail128(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # # Tune5Tail256
    print("New in: Tune5Tail256")
    model = Tune5Tail256(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # Tune5Tail1024
    print("New in: Tune5Tail1024")
    model = Tune5Tail1024(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)


############# free param constraint

    from models import TuneVanillaFree
    from models import Tune3ContFree, Tune2ContFree, Tune1ContFree
    from models import Tune1ExpaFree, Tune2ExpaFree, Tune3ExpaFree
    from models import Tune2TailFree, Tune3TailFree, Tune4TailFree, Tune5TailFree
    from models import TunePlusFree


    # # # # FreeTuneVanilla
    print("New in: FreeTuneVanilla")
    model = TuneVanillaFree(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)

    # # # # FreeTune1Cont
    print("New in: FreeTune1Cont")
    model = Tune1ContFree(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # FreeTune2Cont
    print("New in: FreeTune2Cont")
    model = Tune2ContFree(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # # FreeTune3Cont
    print("New in: FreeTune3Cont")
    model = Tune3ContFree(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    
    # # # Tune1ExpaFree
    print("New in: Tune1ExpaFree")
    model = Tune1ExpaFree(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # FreeTune2Expa
    print("New in: FreeTune2Expa")
    model = Tune2ExpaFree(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # FreeTune3Expa
    print("New in: FreeTune3Expa")
    model = Tune3ExpaFree(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)

    # # # # # FreeTune5Tail
    print("New in: FreeTune5Tail")
    model = Tune5TailFree(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # FreeTune4Tail
    print("New in: FreeTune4Tail")
    model = Tune4TailFree(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # FreeTune3Tail
    print("New in: FreeTune3Tail")
    model = Tune3TailFree(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)
    # # # # FreeTune2Tail
    print("New in: FreeTune2Tail")
    model = Tune2TailFree(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)



    # # # # FreeTailedWithPinkLinksU
    print("New in: FreeTailedWithPinkLinksU")
    model = TunePlusFree(n_classes=50)
    summary(model.to(torch.device('cuda')), input_shape)    



