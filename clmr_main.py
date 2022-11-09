##### 
# This file was created by Janne Spijkervet, from xxxx and slightly altered to incorporate TUNe and avoid duplicate code when combining the directory with jukemir


import argparse
from gc import callbacks
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger  # newline 1
from torch.utils.data import DataLoader

# Audio Augmentations
from torchaudio_augmentations import (
    RandomApply,
    ComposeMany,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    # PitchShift,
    # Reverb,
)

from clmr.data import ContrastiveDataset
from clmr.datasets import get_dataset
from clmr.evaluation import evaluate

from clmr.modules import ContrastiveLearning, SupervisedLearning
from clmr_utils import yaml_config_hook

from models import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CLMR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # ------------
    # data augmentations
    # ------------
    if args.supervised:
        train_transform = [RandomResizedCrop(n_samples=args.audio_length)]
        num_augmented_samples = 1
    else:
        train_transform = [
            RandomResizedCrop(n_samples=args.audio_length),
            RandomApply([PolarityInversion()], p=args.transforms_polarity),
            RandomApply([Noise()], p=args.transforms_noise),
            RandomApply([Gain()], p=args.transforms_gain),
            RandomApply(
                [HighLowPass(sample_rate=args.sample_rate)], p=args.transforms_filters
            ),
            RandomApply([Delay(sample_rate=args.sample_rate)], p=args.transforms_delay),
            # RandomApply(
            #     # [
            #     #     PitchShift(
            #     #         n_samples=args.audio_length,
            #     #         sample_rate=args.sample_rate,
            #     #     )
            #     # ],
            #     p=args.transforms_pitch,
            # ),
            # RandomApply(
            #     [Reverb(sample_rate=args.sample_rate)], p=args.transforms_reverb
            # ),
        ]
        num_augmented_samples = 2

    # ------------
    # dataloaders
    # ------------
    # train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    if args.dataset == "magnatagatune":
        train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    else:
        train_dataset = get_dataset(args.dataset, args.dataset_dir, subset=None)
    # valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")
    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, args.audio_length),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )

    # contrastive_valid_dataset = ContrastiveDataset(
    #     valid_dataset,
    #     input_shape=(1, args.audio_length),
    #     transform=ComposeMany(
    #         train_transform, num_augmented_samples=num_augmented_samples
    #     ),
    # )

    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True,
    )

    # valid_loader = DataLoader(
    #     contrastive_valid_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.workers,
    #     drop_last=True,
    #     shuffle=False,
    # )

    # ------------
    # encoder
    # ------------
    if args.model != "clmr":
        encoder = eval(str(args.model))(n_classes=train_dataset.n_classes)
    else:
        encoder = SampleCNN(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
            supervised=args.supervised,
            out_dim=train_dataset.n_classes,
        )

    # ------------
    # model
    # ------------

    if args.supervised:
        module = SupervisedLearning(args, encoder, output_dim=train_dataset.n_classes)
    else:
        module = ContrastiveLearning(args, encoder)

    # logger = TensorBoardLogger("runs", name="CLMRv2-{}".format(args.dataset))
    logger = WandbLogger(save_dir="runs", name="ISMIR-{}-{}".format(args.dataset, args.model))
    if args.checkpoint_path:
        trainer = Trainer.from_argparse_args(
            args,
            logger=logger,
            sync_batchnorm=True,
            max_epochs=args.max_epochs,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            accelerator=args.accelerator,
        )
        # trainer.fit(module, train_loader, valid_loader, ckpt_path=args.checkpoint_path)
        trainer.fit(module, train_loader, ckpt_path=args.checkpoint_path)
    else:
        # ------------
        # training
        # ------------

        if args.supervised:
            early_stopping = EarlyStopping(monitor="Valid/loss", patience=20)
        else:
            early_stopping = None

        trainer = Trainer.from_argparse_args(
            args,
            logger=logger,
            sync_batchnorm=True,
            max_epochs=args.max_epochs,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            accelerator=args.accelerator,
        )
        # trainer.fit(module, train_loader, valid_loader)
        trainer.fit(module, train_loader)
    trainer.save_checkpoint(filepath="./checkpoints/dataset_{}_model_{}_epoch_{}.ckpt".format(module.hparams.dataset, module.hparams.model, module.current_epoch))
