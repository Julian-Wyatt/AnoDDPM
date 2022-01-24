# https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/gaussian_diffusion.py
import math
from abc import abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import tqdm.auto


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class PositionalEmbedding(nn.Module):
    # PositionalEmbedding
    """
    Computes Positional Embedding of the timestep
    """

    def __init__(self, dim, scale=1):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    def __init__(self, in_channels, use_conv,out_channels=None):
        super().__init__()
        self.channels = in_channels
        out_channels  = out_channels or in_channels
        if use_conv:
            # downsamples by 1/2
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            assert in_channels == out_channels
            self.downsample = nn.AvgPool2d(kernel_size=2,stride=2)

    def forward(self, x, time_embed=None):
        assert x.shape[1] == self.channels
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        self.use_conv = use_conv
        # uses upsample then conv to avoid checkerboard artifacts
        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x, time_embed=None):
        assert x.shape[1] == self.channels
        x = F.interpolate(x,scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(self, in_channels, n_heads=1, n_head_channels=-1):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm32(32,self.in_channels)
        if n_head_channels == -1:
            self.num_heads = n_heads
        else:
            assert (
                    in_channels % n_head_channels == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {n_head_channels}"
            self.num_heads = in_channels // n_head_channels

        # query, key, value for attention
        self.to_qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))

    def forward(self, x,time=None):
        b,c,*spatial = x.shape
        x = x.reshape(b,c,-1)
        qkv = self.to_qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x+h).reshape(b,c,*spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, time=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class ResBlock(TimestepBlock):
    def __init__(self,
                 in_channels,
                 time_embed_dim,
                 dropout,
                 out_channels=None,
                 use_conv = False,
                 up=False,
                 down=False
                 ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(in_channels,False)
            self.x_upd = Upsample(in_channels,False)
        elif down:
            self.h_upd = Downsample(in_channels, False)
            self.x_upd = Downsample(in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.embed_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channels,out_channels,3,padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_embed):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1],self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.embed_layers(time_embed).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNet(nn.Module):
    # UNet model
    def __init__(
            self,
            img_size,
            base_channels,
            conv_resample=True,
            n_heads = 1,
            n_head_channels =-1,
            channel_mults="",
            num_res_blocks =2,
            dropout=0,
            attention_resolutions="32,16,8",
            biggan_updown=True,
    ):
        self.dtype = torch.float32
        super().__init__()

        if channel_mults == "":
            if img_size == 256:
                channel_mults = (1, 1, 2, 2, 4, 4)
            elif img_size == 128:
                channel_mults = (1, 1, 2, 3, 4)
            elif img_size == 64:
                channel_mults = (1, 2, 3, 4)
            elif img_size == 32:
                channel_mults = (1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {img_size}")
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(img_size//int(res))

        self.image_size = img_size
        self.in_channels = 1
        self.model_channels = base_channels
        self.out_channels = 1
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mults
        self.conv_resample = conv_resample

        self.dtype = torch.float32
        self.num_heads = n_heads
        self.num_head_channels = n_head_channels



        time_embed_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            PositionalEmbedding(base_channels, 1),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )


        ch  = int(channel_mults[0] * base_channels)
        self.down = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(self.in_channels, base_channels, 3, padding=1))])
        channels = [ch]
        ds=1
        for i, mult in enumerate(channel_mults):
            # out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                layers = [ResBlock(
                    ch,
                    time_embed_dim=time_embed_dim,
                    out_channels = base_channels * mult,
                    dropout = dropout,
                )]
                ch = base_channels * mult
                # channels.append(ch)

                if ds in attention_ds:
                    layers.append(
                        AttentionBlock(
                            ch,
                            n_heads=n_heads,
                            n_head_channels=n_head_channels,
                        )
                    )
                self.down.append(TimestepEmbedSequential(*layers))
                channels.append(ch)
            if i != len(channel_mults) - 1:
                out_channels = ch
                self.down.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim=time_embed_dim,
                            out_channels=out_channels,
                            dropout=dropout,
                            down=True
                        )
                        if biggan_updown
                        else
                        Downsample(ch, conv_resample, out_channels=out_channels)
                    )
                )
                ds*=2
                ch = out_channels
                channels.append(ch)

        self.middle = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim=time_embed_dim,
                dropout=dropout
            ),
            AttentionBlock(
                ch,
                n_heads=n_heads,
                n_head_channels=n_head_channels
            ),
            ResBlock(
                ch,
                time_embed_dim=time_embed_dim,
                dropout=dropout
            )
        )
        self.up = nn.ModuleList([])

        for i, mult in reversed(list(enumerate(channel_mults))):
            for j in range(num_res_blocks + 1):
                inp_chs = channels.pop()
                layers = [
                    ResBlock(
                        ch+inp_chs,
                        time_embed_dim=time_embed_dim,
                        out_channels=base_channels * mult,
                        dropout=dropout
                    )
                ]
                ch = base_channels * mult
                if ds in attention_ds:
                    layers.append(
                        AttentionBlock(
                            ch,
                            n_heads=n_heads,
                            n_head_channels=n_head_channels
                        ),
                    )

                if i and j == num_res_blocks:
                    out_channels = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim=time_embed_dim,
                            out_channels=out_channels,
                            dropout=dropout,
                            up=True
                        )
                        if biggan_updown
                        else
                        Upsample(ch, conv_resample, out_channels=out_channels)
                    )
                    ds//=2
                self.up.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            GroupNorm32(32,ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(base_channels*channel_mults[0], self.out_channels, 3, padding=1))
        )

    def forward(self, x, time):

        time_embed = self.time_embedding(time)

        skips = []

        h = x.type(self.dtype)
        for i,module in enumerate(self.down):
            h = module(h, time_embed)
            skips.append(h)
        h = self.middle(h,time_embed)
        for i,module in enumerate(self.up):
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, time_embed)
        h = h.type(x.dtype)
        h = self.out(h)
        return h


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def update_ema_params(target,source,decay_rate=0.9999):
    targParams = dict(target.named_parameters())
    srcParams = dict(source.named_parameters())
    for k in targParams:
        targParams[k].data.mul_(decay_rate).add_(srcParams[k].data,alpha=1-decay_rate)


def get_beta_schedule(num_diffusion_steps, name="cosine"):
    betas = []
    if name == "cosine":
        max_beta = 0.999
        f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)
    elif name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas


def extract(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape).to(device)


def mean_flat(tensor):
    return torch.mean(tensor, dim=list(range(1,len(tensor.shape))))


class GaussianDiffusion:
    def __init__(
            self,
            img_size,
            betas,
            img_channels=1,
            loss_type="l2",
            loss_weight='none', #prop t / uniform / None
            ):
        super().__init__()

        self.img_size = img_size
        self.img_channels = img_channels
        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps,0,-1)
        elif loss_weight =="uniform":
            self.weights = np.ones(self.num_timesteps)

        self.loss_weight = loss_weight
        alphas = 1 - betas
        self.betas = betas
        self.sqrt_minus_one_betas = np.sqrt(1-betas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0,self.alphas_cumprod[:-1])
        # self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:],0.0)


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )


    def sample_t_with_weights(self,b_size,device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p),size=b_size,p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1/len(p)*p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

    def predict_x_0_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t,x_t.shape, x_t.device) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)*eps)


    def p_mean_variance(self,model,x_t,t):
        model_output = model(x_t,t)

        model_var = np.append(self.posterior_variance[1],self.betas[1:])
        model_logvar = np.log(model_var)
        model_var = extract(model_var, t, x_t.shape, x_t.device)
        model_logvar = extract(model_logvar, t, x_t.shape, x_t.device)

        pred_x_0 = self.predict_x_0_from_eps(x_t.clamp(-1,1), t, model_output)
        model_mean, _, _ = self.q_posterior_mean_variance(
            pred_x_0, x_t, t
        )
        return {
            "mean": model_mean,
            "variance": model_var,
            "log_variance": model_logvar,
            "pred_x_0": pred_x_0,
        }

    def sample_p(self,model,x_t, t):
        out = self.p_mean_variance(model,x_t,t)
        noise = torch.randn_like(x_t)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5*out["log_variance"])*noise
        return {"sample":sample,"pred_x_0":out["pred_x_0"]}

    def q_posterior_mean_variance(self,x_0, x_t, t):
        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device)*x_t)

        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def forward_backward(self,model, x, see_whole_sequence="half",t_distance=None):
        assert see_whole_sequence == "whole" or see_whole_sequence == "half" or see_whole_sequence == None

        if t_distance is None:
            t_distance = self.num_timesteps
        seq = [x.cpu().detach()]
        if see_whole_sequence == "whole":

            for t in range(1,int(t_distance)):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                noise = torch.randn_like(x)
                with torch.no_grad():
                    x = self.sample_q_gradual(x, t_batch, noise)

                seq.append(x.cpu().detach())
        else:
            x = self.sample_q(x,torch.tensor([t_distance], device=x.device).repeat(x.shape[0]),torch.randn_like(x))
            if see_whole_sequence=="half":
                seq.append(x.cpu().detach())

        for t in range(int(t_distance) -1, -1, -1):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                out = self.sample_p(model,x,t_batch)
                x = out["sample"]
            if see_whole_sequence:
                seq.append(x.cpu().detach())

        return x.cpu().detach() if not see_whole_sequence else seq


    def sample_q(self, x, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x.shape, x.device) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape, x.device) * noise)

    #TODO: curious whether noise needs to be the same across every t here
    def sample_q_gradual(self, x, t, noise):
        return (extract(self.sqrt_minus_one_betas, t, x.shape, x.device) * x +
                extract(self.sqrt_betas, t, x.shape, x.device) * noise)

    def calc_loss(self, model, x, t):
        noise = torch.randn_like(x)

        noisy_x = self.sample_q(x, t, noise)
        estimate_noise = model(noisy_x, t)

        if self.loss_type == "l1":
            loss = mean_flat((estimate_noise - noise).abs())
        elif self.loss_type == "l2":
            loss = mean_flat((estimate_noise - noise).square())
        return loss, noisy_x, estimate_noise


    def p_loss(self, model, x, args):
        if self.loss_weight == "none":
            if args["train_start"]:
                t,weights = torch.randint(0, min(args["sample_distance"]*2,self.num_timesteps), (x.shape[0],), device=x.device),1
            else:
                t, weights = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device), 1
        else:
            t, weights = self.sample_t_with_weights(x.shape[0],x.device)

        loss = self.calc_loss(model, x, t)
        loss = ((loss[0]*weights).mean(),loss[1],loss[2])
        return loss

