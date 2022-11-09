#####
# A combination of CLMR linear evaluation and step 3 of https://github.com/p-lambda/jukemir 
#####

ENCODER_CHECKPOINT_PATH_TAIL = os.path.join(os.getcwd(),"checkpoints/tune_5_tail_epoch_10000.ckpt")
ENCODER_CHECKPOINT_PATH_PLUS = os.path.join(os.getcwd(),"checkpoints/tune_plus_epoch_10000.ckpt")

SAMPLE_RATE = 22050
FRAME_LENGTH = 59049

if __name__ == "__main__":
    import pathlib
    from argparse import ArgumentParser
    
    import os
    import librosa
    import numpy as np
    import torch
    from models import TunePlus, Tune5Tail
    from clmr_utils import load_encoder_checkpoint, Identity
    from tqdm import tqdm

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--batch_idx", type=int)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--model", type=str, default='Tune5Tail', choices=['Tune5Tail','TunePlus'])
    parser.add_argument("--data", type=str, default="giant", choices=['emomusic', 'gtzan','giantsteps','magnatagatune'])


    parser.set_defaults(
        batch_size=None,
        batch_idx=None,
    )
    args = parser.parse_args()

    if type(args.checkpoint) == type(None):
        checkpoint = ENCODER_CHECKPOINT_PATH_TAIL if args.model == "Tune5Tail" else ENCODER_CHECKPOINT_PATH_PLUS

    if args.data == "emomusic":
        input_dir = pathlib.Path(os.path.join(os.getcwd(),".jukemir/processed/emomusic/wav"))
        output_dir = pathlib.Path(os.path.join(os.getcwd(),f".jukemir/representations/emomusic/{args.model}/"))
    elif args.data == "gtzan":
        input_dir = pathlib.Path(os.path.join(os.getcwd(),".jukemir/processed/gtzan_ff/wav"))
        output_dir = pathlib.Path(os.path.join(os.getcwd(),f".jukemir/representations/gtzan_ff/{args.model}/"))
    elif args.data == "giantsteps":
        input_dir = pathlib.Path(os.path.join(os.getcwd(),".jukemir/processed/giantsteps_clips/wav"))
        output_dir = pathlib.Path(os.path.join(os.getcwd(),f".jukemir/representations/giantsteps_clips/{args.model}/"))
    elif args.data == "magnatagatune":
        input_dir = pathlib.Path(os.path.join(os.getcwd(),".jukemir/processed/magnatagatune/wav"))
        output_dir = pathlib.Path(os.path.join(os.getcwd(),f".jukemir/representations/magnatagatune/{args.model}/"))



    output_dir.mkdir(exist_ok=True)
    input_paths = sorted(list(input_dir.iterdir()))
    if args.batch_size is not None and args.batch_idx is not None:
        batch_starts = list(range(0, len(input_paths), args.batch_size))
        if args.batch_idx >= len(batch_starts):
            raise ValueError("Invalid batch index")
        batch_start = batch_starts[args.batch_idx]
        input_paths = input_paths[batch_start : batch_start + args.batch_size]

    encoder = None
    for input_path in tqdm(input_paths):
        # Check if output already exists
        output_path = pathlib.Path(output_dir, f"{input_path.stem}.npy")
        try:
            np.load(output_path)
            continue
        except:
            pass

        if encoder is None:
            # References:
            # - https://colab.research.google.com/drive/1Njz8EoN4br587xjpRKcssMuqQY6Cc5nj#scrollTo=igc0TggNyj8U
            # - https://github.com/Spijkervet/CLMR/blob/master/linear_evaluation.py
            # - https://github.com/Spijkervet/CLMR/blob/0e52a20c7687ecec00c4d223230f00bffe7430a7/clmr/evaluation.py#L8

            encoder = eval(args.model)(
                n_classes=50,
            )

            state_dict = load_encoder_checkpoint(args.checkpoint, 50)
            encoder.load_state_dict(state_dict)
            encoder.fc = Identity()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder.eval()
            encoder.to(device)

        with torch.no_grad():
            audio, sr = librosa.core.load(input_path, sr=SAMPLE_RATE)
            if audio.ndim == 1:
                audio = audio[np.newaxis]
            audio = torch.tensor(audio, dtype=torch.float32, device=device)
            audio = torch.mean(audio, axis=0, keepdim=True)
            frames = torch.split(audio, FRAME_LENGTH, dim=1)
            if len(frames) <= 1:
                raise Exception("Audio too short")
            frames = torch.cat(frames[:-1], dim=0)
            frames = frames.unsqueeze(dim=1)
            representations = encoder(frames)
            representation = representations.mean(dim=0).cpu().numpy()

        np.save(output_path, representation)