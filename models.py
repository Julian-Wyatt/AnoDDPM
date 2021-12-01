# https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/gaussian_diffusion.py


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import tqdm.auto

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
    def __init__(self, in_channels):
        super().__init__()
        # downsamples by 1/2
        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_embed=None):
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # uses upsample then conv to avoid checkerboard artifacts
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1))

    def forward(self, x, time_embed=None):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    # https://github.com/abarankab/DDPM/blob/387eb1e1e2bcc4226294429412107b570c66ec9a/ddpm/unet.py#L214
    # https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    def __init__(self, in_channels, num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups, self.in_channels)
        # query, key, value for attention
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        # initial Mat mult * 1/sqrt(c)
        mat_mult = torch.bmm(q, k) * (c ** (-0.5))
        assert mat_mult.shape == (b, h * w, w * h)

        attention = torch.softmax(mat_mult, dim=-1)
        output = torch.bmm(attention, v)
        assert output.shape == (b, h * w, c)

        output = output.view(b, h, w, c).permute(0, 3, 1, 2)
        return self.out(output) + x


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout,
                 activation=F.relu,
                 num_groups=32,
                 time_embed_dim=256,
                 use_attention=False):
        super().__init__()
        self.activation = activation

        self.norm_in = nn.GroupNorm(num_groups, in_channels)
        self.conv_in = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_bias = nn.Linear(time_embed_dim, out_channels) if time_embed_dim is not None else None

        self.norm_out = nn.GroupNorm(num_groups, out_channels)
        self.conv_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.residual_connection = nn.Conv2d(in_channels, out_channels, 1)\
            if in_channels != out_channels else nn.Identity()
        self.attention = AttentionBlock(out_channels, num_groups) if use_attention else nn.Identity()


    def forward(self, x, time_embed=None):
        out = self.activation(self.norm_in(x))
        out = self.conv_in(out)

        if self.time_bias:
            out += self.time_bias(self.activation(time_embed))[:, :, None, None]

        out = self.activation(self.norm_out(out))
        out = self.conv_out(out) + self.residual_connection(x)
        return self.attention(out)


class UNet(nn.Module):

    # UNet model
    def __init__(
            self,
            img_size,
            base_channels,
            channel_mults="",
            res_block_layer_count=2,
            activation=F.relu,
            dropout=0.2,
            attention_resolutions=(None, 32, 16, 8),
            time_embed_dim=256,
            time_embed_scale=1,
            num_groups=32):
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

        self.activation = activation
        in_channels = 1
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.time_embedding = nn.Sequential(
            PositionalEmbedding(base_channels, time_embed_scale),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),  # sigmoid linear unit
            nn.Linear(time_embed_dim, time_embed_dim)
        ) if time_embed_dim is not None else None

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        channels = [base_channels]
        cur_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(res_block_layer_count):
                self.down.append(ResBlock(
                    cur_channels,
                    out_channels,
                    dropout,
                    activation=activation,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                cur_channels = out_channels
                channels.append(cur_channels)

            if i != len(channel_mults) - 1:
                self.down.append(Downsample(cur_channels))
                channels.append(cur_channels)

        self.bottom = nn.ModuleList([ResBlock(cur_channels,
                                              cur_channels,
                                              dropout,
                                              activation,
                                              num_groups,
                                              use_attention=True),
                                     ResBlock(cur_channels,
                                              cur_channels,
                                              dropout,
                                              activation,
                                              num_groups,
                                              use_attention=False)
                                     ])

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(res_block_layer_count + 1):
                self.up.append(ResBlock(
                    channels.pop() + cur_channels,
                    out_channels,
                    dropout,
                    activation=activation,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                cur_channels = out_channels

            if i != 0:
                self.up.append(Upsample(cur_channels))
        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        self.norm_out = nn.GroupNorm(num_groups, base_channels)

    def forward(self, x, time=None):

        time_embed = self.time_embedding(time) if self.time_embedding else None

        x = self.conv_in(x)
        skips = [x]

        for i in self.down:
            x = i(x, time_embed)
            skips.append(x)

        for i in self.bottom:
            x = i(x, time_embed)

        for i in self.up:
            if isinstance(i, ResBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = i(x, time_embed)
        x = self.activation(self.norm_out(x))
        x = self.conv_out(x)

        return x


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


class GaussianDiffusion(nn.Module):
    def __init__(
            self, model,
            img_size,
            betas,
            img_channels=1,
            loss_type="l2",
            loss_weight='prop-t',
            eta=0):
        super().__init__()

        self.model = model
        self.img_size = img_size
        self.img_channels = img_channels
        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps,0,-1)
        else:
            self.weights = np.ones(self.num_timesteps)
        alphas = 1 - betas
        self.betas = betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0,self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:],0.0)


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
        self.eta = eta

    def sample_t_with_weights(self,b_size,device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p),size=b_size,p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1/len(p)*p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

    @torch.no_grad()
    def predict_x_0_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t,x_t.shape, x_t.device) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)*eps)


    @torch.no_grad()
    def p_mean_variance(self,x_t,t):
        model_output = self.model(x_t,t)

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

    def sample_p(self,x_t, t):
        out = self.p_mean_variance(x_t,t)
        noise = torch.randn_like(x_t)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5*out["log_variance"])*noise
        return {"sample":sample.clamp(-1,1),"pred_x_0":out["pred_x_0"].clamp(-1,1)}

    def sample_p_ddim(self,x_t,t,eta):
        out = self.p_mean_variance(x_t,t)
        eps = self.model(x_t,t)
        alpha_bar = extract(self.alphas_cumprod,t,x_t.shape,x_t.device)
        alpha_bar_prev = extract(self.alphas_cumprod_prev,t,x_t.shape,x_t.device)
        # experiment with eta 0-1
        sigma = (
            eta *
            torch.sqrt((1-alpha_bar_prev)/(1-alpha_bar)) *
            torch.sqrt(1-alpha_bar/alpha_bar_prev)
        )

        #Equation 12 from DDIM
        mean_pred = (
            torch.sqrt(alpha_bar_prev) * out['pred_x_0'] + torch.sqrt(1-alpha_bar_prev-sigma**2)*eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )
        noise = torch.randn_like(x_t)
        sample = mean_pred + nonzero_mask * sigma*noise
        return {'sample':sample, 'pred_x_0':out['pred_x_0']}


    @torch.no_grad()
    def q_posterior_mean_variance(self,x_0, x_t, t):
        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device)*x_t)

        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped

    @torch.no_grad()
    def forward_backward(self, x, see_whole_sequence=False,t_distance=None, use_ddim=False):
        if t_distance is None:
            t_distance = self.num_timesteps

        if see_whole_sequence:
            seq = [x.cpu().detach()]

        for t in range(int(t_distance)):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            noise = torch.randn_like(x)
            x = self.sample_q(x, t_batch, noise)

            if see_whole_sequence:
                seq.append(x.cpu().detach())
        if use_ddim:
            for t in range(self.num_timesteps - 1, -1, -1):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                with torch.no_grad():
                    out = self.sample_p_ddim(x, t_batch, self.eta)
                    x = out["sample"]
                if see_whole_sequence:
                    seq.append(x.cpu().detach())
        else:
            for t in range(int(t_distance) - 1, -1, -1):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                with torch.no_grad():
                    out = self.sample_p(x,t_batch)
                    x = out["sample"]
                if see_whole_sequence:
                    seq.append(x.cpu().detach())

        return x.cpu().detach() if not see_whole_sequence else seq

    @torch.no_grad()
    def sample(self, x, see_whole_sequence=False):

        if see_whole_sequence:
            seq = [x.cpu().detach()]

        with tqdm.tqdm(int(self.num_timesteps) + int(self.num_timesteps)) as pbar:

            for t in range(int(self.num_timesteps) - 1, -1, -1):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                with torch.no_grad():
                    out = self.sample_p(x, t_batch)
                    x = out["sample"]
                pbar.update(1)
                if see_whole_sequence:
                    seq.append(x.cpu().detach())

        return x.cpu().detach() if not see_whole_sequence else seq

    def sample_q(self, x, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x.shape, x.device) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape, x.device) * noise)

    def calc_loss(self, x, t):
        noise = torch.randn_like(x)

        noisy_x = self.sample_q(x, t, noise)
        estimate_noise = self.model(noisy_x, t)

        if self.loss_type == "l1":
            loss = mean_flat((estimate_noise - noise).abs())
        elif self.loss_type == "l2":
            loss = mean_flat((estimate_noise - noise).square())
        return loss, noisy_x, estimate_noise



    def forward(self, x):
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        t, weights = self.sample_t_with_weights(x.shape[0],x.device)

        loss = self.calc_loss(x, t)
        loss = ((loss[0]*weights).mean(),loss[1],loss[2])
        # print(weights,loss[0])
        return loss
