from functools import wraps

import numpy as np

import torch
import torch
import torch.nn as nn
import e3nn.o3 as o3
import e3nn.nn as enn
from einops import rearrange
from torch import einsum
from torch_cluster import fps
import re
import torch.nn.functional as F

from timm.models.layers import DropPath

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

import torch
import torch.nn as nn
from e3nn import o3


class EquivariantPreNorm(nn.Module):
    def __init__(self, irreps, fn, context_irreps=None, eps=1e-5):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.nch_s = self.irreps.count("0e")
        self.nch_v = self.irreps.count("1o")
        self.eps = eps
        self.norm_context = EquivariantPreNorm(context_irreps, fn=nn.Identity(), eps=eps) if exists(context_irreps) else None
        self.fn = fn

    def forward(self, x, **kwargs):
        B, N = x.shape[:2]
        x_s = x[..., :self.nch_s].reshape(B, N, self.nch_s)
        x_v = x[..., self.nch_s:].reshape(B, N, self.nch_v, 3)        # B, N, nch_v, 3

        mu_s = x_s.mean(dim=-1, keepdim=True)
        x_s = x_s - mu_s
        var_s = x_s.pow(2).mean(dim=-1, keepdim=True)
        x_s = x_s / torch.sqrt(var_s + self.eps)

        mu_v = x_v.mean(dim=-2, keepdim=True)                 # [B, N, 1, 3]
        x_v = x_v - mu_v
        var_v = x_v.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True) / 3  # [B, N, 1]
        x_v = x_v / torch.sqrt(var_v.unsqueeze(-1) + self.eps)

        x_v = x_v.reshape(B, N, -1)
        x = torch.cat([x_s, x_v], dim=-1)

        if exists(self.norm_context):
            normed_context = self.norm_context(kwargs['k_feats'])
            kwargs.update(k_feats = normed_context)
        return self.fn(x, **kwargs)


class EquivariantFeedForward(nn.Module):
  def __init__(self, dim="32x0e + 16x1o", mul=2):
    super().__init__()
    dim_hid  = re.sub(r'(\d+)x', lambda m: f"{int(m.group(1)) * mul}x",dim)
    self.irreps_in = o3.Irreps(dim)
    self.irreps_hid   = o3.Irreps(dim_hid)

    num_s = self.irreps_hid.count("0e")
    num_v = self.irreps_hid.count("1o")

    irreps_scalars = o3.Irreps(f"{num_s}x0e")
    act_scalars = [torch.nn.functional.silu]

    irreps_gates = o3.Irreps(f"{num_v}x0e")
    act_gates = [torch.sigmoid]

    irreps_gated = o3.Irreps(f"{num_v}x1o")

    self.gate = enn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=act_scalars,
            irreps_gates=irreps_gates,
            act_gates=act_gates,
            irreps_gated=irreps_gated
        )
    self.fc1 = o3.Linear(self.irreps_in, self.gate.irreps_in)
    self.fc2 = o3.Linear(self.gate.irreps_out, self.irreps_in)

  def forward(self, x):
    x = self.fc1(x)
    x = self.gate(x)
    x = self.fc2(x)
    return x
  
class EquivariantAttention(nn.Module):
    def __init__(self, irreps_dim="128x0e + 64x1o"):
        super().__init__()
        self.irreps  = o3.Irreps(irreps_dim)

        self.scale = (self.irreps.dim) ** -0.5
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=1)

        self.to_q = o3.Linear(self.irreps, self.irreps)

        self.to_k = o3.FullyConnectedTensorProduct(
            self.irreps,
            self.irreps_sh,
            self.irreps,
            shared_weights=True,
        )
        self.fc_k = nn.Sequential(
                nn.Linear(1, 32),
                nn.SiLU(),
                nn.Linear(32, 1)
            )

        self.to_v = o3.FullyConnectedTensorProduct(
            self.irreps,
            self.irreps_sh,
            self.irreps,
            shared_weights=True,
        )
        self.fc_v = nn.Sequential(
                nn.Linear(1, 32),
                nn.SiLU(),
                nn.Linear(32, 1)
            )

        self.dot = o3.FullyConnectedTensorProduct(self.irreps, self.irreps, "0e")

    def invariant_dot(self, q, k):
        ns = self.irreps.count("0e")
        nv = self.irreps.count("1o")

        q_s = q[..., :ns]
        k_s = k[..., :ns]

        scalar_sim = (q_s * k_s).sum(dim=-1)

        if nv > 0:
            q_v = q[..., ns:].reshape(*q.shape[:-1], nv, 3)
            k_v = k[..., ns:].reshape(*k.shape[:-1], nv, 3)

            vector_sim = (q_v * k_v).sum(dim=(-1, -2))
        else:
            vector_sim = 0.0

        return scalar_sim + vector_sim

    def forward(self, q_feats, k_feats, x_rel, dist):
        B, M, _ = q_feats.shape
        N = k_feats.shape[1]

        sh = o3.spherical_harmonics(self.irreps_sh, x_rel, normalize=False)
        w_k = self.fc_k(dist)
        w_v = self.fc_v(dist)


        k_feats = k_feats.unsqueeze(1).expand(-1, M, -1, -1)          # [B, M, N, irreps_dim]

        q = self.to_q(q_feats).unsqueeze(2).expand(-1, -1, N, -1)     # [B, M, N, irreps_dim]
        k = self.to_k(k_feats, sh)                                    # [B, M, N, irreps_dim]
        v = self.to_v(k_feats, sh)                                    # [B, M, N, irreps_dim]

        k = k * w_k
        v = v * w_v

        sim =  self.invariant_dot(q, k) * self.scale

        attn = torch.softmax(sim, dim=-1)

        h_out = torch.einsum('b m n, b m n d -> b m d', attn, v)

        return h_out

class EquivariantPointEmbed(nn.Module):
    def __init__(self, irreps_dim="128x0e + 64x1o"):
        super().__init__()

        self.irreps_out = o3.Irreps(irreps_dim)
        self.num_scalars = self.irreps_out.count("0e")
        self.num_vectors = self.irreps_out.count("1o")

        self.type0Vector = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, self.num_scalars)
        )

        self.type1Vector = nn.Parameter(torch.ones(1, 1, self.num_vectors, 1))
        self.fc = o3.Linear(self.irreps_out, self.irreps_out)

    def forward(self, x):
        dist = torch.norm(x, dim=-1, keepdim=True)
        dist = dist.clone()
        dist[dist == 0] = 1.0
        r_hat = x / dist

        f_0 = self.type0Vector(dist)
        f_1 = r_hat.unsqueeze(2) * self.type1Vector            # B, N, num_vectors, 3

        f_1 = rearrange(f_1, 'b n c m -> b n (c m)', m=3)
        feats = torch.cat([f_0, f_1], dim=-1)
        return self.fc(feats)

class EquivariantAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=5,
        irreps_dim="64x0e + 64x1o",
        num_inputs = 512,
        num_latents = 512,
    ):
        super().__init__()

        self.irreps = o3.Irreps(irreps_dim)
        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.point_embed = EquivariantPointEmbed(irreps_dim=irreps_dim)

        self.cross_attend_blocks = nn.ModuleList([
            EquivariantPreNorm(irreps_dim, EquivariantAttention(irreps_dim=irreps_dim), context_irreps=irreps_dim),
            EquivariantPreNorm(irreps_dim, EquivariantFeedForward(dim=irreps_dim, mul=2))
        ])

        self.layers = nn.ModuleList([
            nn.ModuleList([
                EquivariantPreNorm(
                    irreps_dim,
                    EquivariantAttention(irreps_dim=irreps_dim)
                ),
                EquivariantPreNorm(
                    irreps_dim,
                    EquivariantFeedForward(dim=irreps_dim, mul=2)
                )
            ])
            for _ in range(depth)
        ])

        self.dec_cross_attn = EquivariantPreNorm(irreps_dim, EquivariantAttention(irreps_dim=irreps_dim), context_irreps=irreps_dim)
        self.dec_cross_ff = EquivariantPreNorm(irreps_dim, EquivariantFeedForward(dim=irreps_dim, mul=2))

        self.to_outputs = o3.Linear(o3.Irreps(irreps_dim), o3.Irreps("1x0e"))

    @staticmethod
    def compute_geometry(src_pts, dst_pts):
        diff = src_pts.unsqueeze(2) - dst_pts.unsqueeze(1)  # [B, M, N, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist = dist.clone()
        dist[dist == 0] = 1.0
        x_rel = diff / dist
        return x_rel, dist

    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs

        ###### fps
        flattened = pc.view(B*N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        ratio = 1.0 * self.num_latents / self.num_inputs

        idx = fps(pos, batch, ratio=ratio, random_start=False)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc)
        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks
        x_rel, dist = self.compute_geometry(src_pts=sampled_pc, dst_pts=pc)

        x = cross_attn(sampled_pc_embeddings, k_feats=pc_embeddings, 
                       x_rel=x_rel, 
                       dist=dist) + sampled_pc_embeddings
        x = cross_ff(x) + x

        return x, sampled_pc


    def decode(self, x, sampled_pc, queries):

        x_rel_latent, dist_latent = self.compute_geometry(src_pts=sampled_pc, dst_pts=sampled_pc)
        for self_attn, self_ff in self.layers:
            x = self_attn(x, k_feats=x, 
                        x_rel=x_rel_latent, 
                        dist=dist_latent) + x
            x = self_ff(x) + x

        queries_embeddings = self.point_embed(queries)
        x_rel_q, dist_q = self.compute_geometry(src_pts=queries, dst_pts=sampled_pc)

        latents = self.dec_cross_attn(queries_embeddings, k_feats=x, 
                                      x_rel=x_rel_q, 
                                      dist=dist_q) + queries_embeddings
        latents = self.dec_cross_ff(latents) + latents

        out_logits = self.to_outputs(latents)
        return out_logits

    def forward(self, pc, queries, return_latents=False):
        latents, sampled_pc = self.encode(pc)

        o = self.decode(latents, sampled_pc, queries).squeeze(-1)

        if return_latents:
            return {'logits': o, 
                    'latents': latents}
        else:
            return {'logits': o}
        
def create_autoencoder(irreps_dim="128x0e + 64x1o", M=512, N=2048, determinisitc=True):
    if determinisitc:
        model = EquivariantAutoEncoder(
            irreps_dim=irreps_dim,
            num_inputs=N,
            num_latents=M
        )
    return model

###
def ae_d512_m256(N=512):
    return create_autoencoder(irreps_dim="128x0e + 128x1o", M=256, N=N, determinisitc=True)

### Reduced version 
def ae_d256_m128(N=256):
    return create_autoencoder(irreps_dim="64x0e + 64x1o", M=128, N=N, determinisitc=True)