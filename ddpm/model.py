from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d=32, hidden_dim=128):
        super().__init__()
        assert d % 2 == 0, "Sinusoidal embedding dimension d must be even"
        self.d = d
        self.proj = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, t):
        # t: (B,)
        t = t.float()
        device = t.device

        half_dim = self.d // 2

        freqs = torch.exp(
            torch.arange(half_dim, device=device).float()
            * -(torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1))
        )

        emb = t[:, None] * freqs[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return self.proj(emb)  # (B, hidden_dim)

class TransformerBlock2d(nn.Module):
    def __init__(self, channels, num_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, ff_mult * channels),
            nn.SiLU(),
            nn.Linear(ff_mult * channels, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # Sub-layer 1: self-attention with residual
        attn_in = self.attn_norm(x_seq)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x_seq = x_seq + attn_out

        # Sub-layer 2: feed-forward with residual
        ff_in = self.ff_norm(x_seq)
        ff_out = self.ff(ff_in)
        x_seq = x_seq + ff_out

        x = x_seq.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x

class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, num_groups=32):
        super().__init__()
        
        # Handle case where in_ch < 32
        num_groups = min(num_groups, in_ch, out_ch)
        
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch),
            )
            # zero init for stable early training
            nn.init.zeros_(self.time_mlp[-1].weight)
            nn.init.zeros_(self.time_mlp[-1].bias)
        else:
            self.time_mlp = None
        
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )
    
    def forward(self, x, t_emb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.skip(x)  # Residual connection

class UNet(nn.Module):

    def __init__(self, channels: List[int] = [32, 64, 128], convs_per_level=2,
             kernel_size=3, pool_size=2, padding=1, num_heads_att=4,
             time_emb_dim=128, time_emb_base_dim=32):
        super().__init__()

        full_channels = [1] + channels + channels[-2::-1]
        self.mid = len(full_channels) // 2

        self.encoder_convs = nn.ModuleList([
          nn.ModuleList([
              ResConvBlock(
                  in_ch=full_channels[i] if j == 0 else full_channels[i+1],
                  out_ch=full_channels[i+1],
                  time_emb_dim=time_emb_dim
              )
              for j in range(convs_per_level)
          ])
          for i in range(self.mid)
      ])

        self.decoder_convs = nn.ModuleList([
          nn.ModuleList([
              ResConvBlock(
                  in_ch=(full_channels[self.mid + i] + full_channels[self.mid - i - 1]) if j == 0
                      else full_channels[self.mid + i + 1],
                  out_ch=full_channels[self.mid + i + 1],
                  time_emb_dim=time_emb_dim
              )
              for j in range(convs_per_level)
          ])
          for i in range(len(full_channels) - self.mid - 1)
      ])

        #time embedding
        if time_emb_dim is None:
            self.time_embedding = None
        else:
            self.time_embedding = SinusoidalTimeEmbedding(
                d=time_emb_base_dim,
                hidden_dim=time_emb_dim
            )
   
        #attention
        bottleneck_channels = full_channels[self.mid]
    
        if num_heads_att < 2:
            self.attention = nn.Identity()
        else:
            assert bottleneck_channels % num_heads_att == 0, (
                f"number of heads for attention is invalid"
            )
            self.attention = TransformerBlock2d(
              bottleneck_channels,
              num_heads=num_heads_att,
              ff_mult=4,
              dropout=0.0,
              )

        self.output_conv = nn.Conv2d(full_channels[-1], 1, kernel_size=1)
        self.pool = nn.MaxPool2d(pool_size)
        self.upsample = nn.Upsample(scale_factor=pool_size)
        self.activation_fn = F.relu

        self.config = {
            'channels':        channels,
            'convs_per_level': convs_per_level,
            'kernel_size':     kernel_size,
            'pool_size':       pool_size,
            'padding':         padding,
            'num_heads_att':   num_heads_att,
            'time_emb_dim':    time_emb_dim,
            'time_emb_base_dim': time_emb_base_dim,
        }

    def forward(self, x, t=None):
        t_emb = None if self.time_embedding is None or t is None else self.time_embedding(t)
      
        skips = []
        last_encoder = len(self.encoder_convs) - 1
      
      # ENCODER
        for i, conv_block in enumerate(self.encoder_convs):
            for block in conv_block:
                x = block(x, t_emb)  
          
            if i < last_encoder:
                skips.append(x)
                x = self.pool(x)
      
      # BOTTLENECK
        x = self.attention(x)
      
      # DECODER
        for i, conv_block in enumerate(self.decoder_convs):
            x = self.upsample(x)
            x = torch.cat([x, skips.pop()], dim=1)
          
            for block in conv_block:
                x = block(x, t_emb)  
      
        return self.output_conv(x)
