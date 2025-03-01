from miditoolkit import MidiFile
import torch
from pathlib import Path
import wandb
from .os import run_command
from .media import save_img

def log_image(img: torch.Tensor, name: str, step: int, save_dir: Path = Path('')):
    '''
    image shape: (c, h, w) or (h, w)
    '''
    save_dir = Path(wandb.run.dir) / save_dir
    save_dir.mkdir(exist_ok=True)
    img_path = save_dir / f"{name}_{step:08}.jpg"
    save_img(img, img_path)
    image = wandb.Image(str(img_path))
    wandb.log({name: image}, step=step)
    return image

def log_midi_as_audio(midi: MidiFile, name: str, step: int, save_dir: Path = Path(''), soundfont_path: Path = Path("./ignore/FluidR3_GM.sf2")):
    save_dir = Path(wandb.run.dir) / save_dir
    save_dir.mkdir(exist_ok=True)
    midi_path = save_dir / f"{name}_{step:08}.mid"
    midi.dump(str(midi_path))

    audio_path = save_dir / f"{name}_{step:08}.wav"
    run_command(f"fluidsynth -F {audio_path} {soundfont_path} {midi_path}")
    audio = wandb.Audio(str(audio_path), caption=f"{name} {step}")
    wandb.log({name: audio}, step=step)
    return audio