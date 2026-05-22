import torch
from .model import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_unet( state_dict_path: str, channels, convs_per_level, num_heads_att=4, time_emb_dim=128, time_emb_base_dim=32,) -> UNet:
    unet = UNet(  channels=channels,  convs_per_level=convs_per_level,  num_heads_att=num_heads_att,  time_emb_dim=time_emb_dim,  time_emb_base_dim=time_emb_base_dim, )
    unet.load_state_dict(torch.load(state_dict_path, map_location=device))
    unet.to(device)
    unet.eval()
    return unet


def channel_list(channel0: int):
    return [channel0, channel0 * 2, channel0 * 4]


def model_name(channel0: int, convs_per_level: int):
    return f'C0_{channel0}_convs_{convs_per_level}'


def path_name(channel0: int, convs_per_level: int, add_desc=""):
    return f"base_{model_name(channel0, convs_per_level)}{add_desc}.pkl"