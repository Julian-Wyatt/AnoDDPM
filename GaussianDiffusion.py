# https://github.com/openai/guided-diffusion/tree/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924
import random

import matplotlib.pyplot as plt
import numpy as np

import evaluation
from helpers import *
from simplex import Simplex_CLASS


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
    return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL Divergence between two gaussians

    :param mean1:
    :param logvar1:
    :param mean2:
    :param logvar2:
    :return: KL Divergence between N(mean1,logvar1^2) & N(mean2,logvar2^2))
    """
    return 0.5 * (-1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretised_gaussian_log_likelihood(x, means, log_scales):
    """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.
        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))

    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
            )
    assert log_probs.shape == x.shape
    return log_probs


def generate_simplex_noise(
        Simplex_instance, x, t, random_param=False, octave=6, persistence=0.8, frequency=64,
        in_channels=1
        ):
    noise = torch.empty(x.shape).to(x.device)
    for i in range(in_channels):
        Simplex_instance.newSeed()
        if random_param:
            param = random.choice(
                    [(2, 0.6, 16), (6, 0.6, 32), (7, 0.7, 32), (10, 0.8, 64), (5, 0.8, 16), (4, 0.6, 16), (1, 0.6, 64),
                     (7, 0.8, 128), (6, 0.9, 64), (2, 0.85, 128), (2, 0.85, 64), (2, 0.85, 32), (2, 0.85, 16),
                     (2, 0.85, 8),
                     (2, 0.85, 4), (2, 0.85, 2), (1, 0.85, 128), (1, 0.85, 64), (1, 0.85, 32), (1, 0.85, 16),
                     (1, 0.85, 8),
                     (1, 0.85, 4), (1, 0.85, 2), ]
                    )
            # 2D octaves seem to introduce directional artifacts in the top left
            noise[:, i, ...] = torch.unsqueeze(
                    torch.from_numpy(
                            # Simplex_instance.rand_2d_octaves(
                            #         x.shape[-2:], param[0], param[1],
                            #         param[2]
                            #         )
                            Simplex_instance.rand_3d_fixed_T_octaves(
                                    x.shape[-2:], t.detach().cpu().numpy(), param[0], param[1],
                                    param[2]
                                    )
                            ).to(x.device), 0
                    ).repeat(x.shape[0], 1, 1, 1)
        noise[:, i, ...] = torch.unsqueeze(
                torch.from_numpy(
                        # Simplex_instance.rand_2d_octaves(
                        #         x.shape[-2:], octave,
                        #         persistence, frequency
                        #         )
                        Simplex_instance.rand_3d_fixed_T_octaves(
                                x.shape[-2:], t.detach().cpu().numpy(), octave,
                                persistence, frequency
                                )
                        ).to(x.device), 0
                ).repeat(x.shape[0], 1, 1, 1)
    return noise


def random_noise(Simplex_instance, x, t):
    param = random.choice(
            ["gauss", "simplex"]
            )
    if param == "gauss":
        return torch.randn_like(x)
    else:
        return generate_simplex_noise(Simplex_instance, x, t)


class GaussianDiffusionModel:
    def __init__(
            self,
            img_size,
            betas,
            img_channels=1,
            loss_type="l2",  # l2,l1 hybrid
            loss_weight='none',  # prop t / uniform / None
            noise="gauss",  # gauss / perlin / simplex
            ):
        super().__init__()
        if noise == "gauss":
            self.noise_fn = lambda x, t: torch.randn_like(x)

        else:
            self.simplex = Simplex_CLASS()
            if noise == "simplex_randParam":
                self.noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, True, in_channels=img_channels)
            elif noise == "random":
                self.noise_fn = lambda x, t: random_noise(self.simplex, x, t)
            else:
                self.noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, False, in_channels=img_channels)

        self.img_size = img_size
        self.img_channels = img_channels
        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps, 0, -1)
        elif loss_weight == "uniform":
            self.weights = np.ones(self.num_timesteps)

        self.loss_weight = loss_weight
        alphas = 1 - betas
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
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


    def sample_t_with_weights(self, b_size, device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p), size=b_size, p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / len(p) * p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

    def predict_x_0_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps)

    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - pred_x_0) \
               / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)

    def q_mean_variance(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape, x_0.device)
        log_variance = extract(
                self.log_one_minus_alphas_cumprod, t, x_0.shape, x_0.device
                )
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """

        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t)

        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(self, model, x_t, t, estimate_noise=None):
        """
        Finds the mean & variance from N(x_{t-1}; mu_theta(x_t,t), sigma_theta (x_t,t))

        :param model:
        :param x_t:
        :param t:
        :return:
        """
        if estimate_noise == None:
            estimate_noise = model(x_t, t)

        # fixed model variance defined as \hat{\beta}_t - could add learned parameter
        model_var = np.append(self.posterior_variance[1], self.betas[1:])
        model_logvar = np.log(model_var)
        model_var = extract(model_var, t, x_t.shape, x_t.device)
        model_logvar = extract(model_logvar, t, x_t.shape, x_t.device)

        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(
                pred_x_0, x_t, t
                )
        return {
            "mean":         model_mean,
            "variance":     model_var,
            "log_variance": model_logvar,
            "pred_x_0":     pred_x_0,
            }

    def sample_p(self, model, x_t, t, denoise_fn="gauss"):
        out = self.p_mean_variance(model, x_t, t)
        # noise = torch.randn_like(x_t)
        if type(denoise_fn) == str:
            if denoise_fn == "gauss":
                noise = torch.randn_like(x_t)
            elif denoise_fn == "noise_fn":
                noise = self.noise_fn(x_t, t).float()
            elif denoise_fn == "random":
                # noise = random_noise(self.simplex, x_t, t).float()
                noise = torch.randn_like(x_t)
            else:
                noise = generate_simplex_noise(self.simplex, x_t, t, False, in_channels=self.img_channels).float()
        else:
            noise = denoise_fn(x_t, t)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}

    def forward_backward(
            self, model, x, see_whole_sequence="half", t_distance=None, denoise_fn="gauss",
            ):
        assert see_whole_sequence == "whole" or see_whole_sequence == "half" or see_whole_sequence == None

        if t_distance == 0:
            return x.detach()

        if t_distance is None:
            t_distance = self.num_timesteps
        seq = [x.cpu().detach()]
        if see_whole_sequence == "whole":

            for t in range(int(t_distance)):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                # noise = torch.randn_like(x)
                noise = self.noise_fn(x, t_batch).float()
                with torch.no_grad():
                    x = self.sample_q_gradual(x, t_batch, noise)

                seq.append(x.cpu().detach())
        else:
            # x = self.sample_q(x,torch.tensor([t_distance], device=x.device).repeat(x.shape[0]),torch.randn_like(x))
            t_tensor = torch.tensor([t_distance - 1], device=x.device).repeat(x.shape[0])
            x = self.sample_q(
                    x, t_tensor,
                    self.noise_fn(x, t_tensor).float()
                    )
            if see_whole_sequence == "half":
                seq.append(x.cpu().detach())

        for t in range(int(t_distance) - 1, -1, -1):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                out = self.sample_p(model, x, t_batch, denoise_fn)
                x = out["sample"]
            if see_whole_sequence:
                seq.append(x.cpu().detach())

        return x.detach() if not see_whole_sequence else seq

    def sample_q(self, x_0, t, noise):
        """
            q (x_t | x_0 )

            :param x_0:
            :param t:
            :param noise:
            :return:
        """
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise)

    def sample_q_gradual(self, x_t, t, noise):
        """
        q (x_t | x_{t-1})
        :param x_t:
        :param t:
        :param noise:
        :return:
        """
        return (extract(self.sqrt_alphas, t, x_t.shape, x_t.device) * x_t +
                extract(self.sqrt_betas, t, x_t.shape, x_t.device) * noise)

    def calc_vlb_xt(self, model, x_0, x_t, t, estimate_noise=None):
        # find KL divergence at t
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_0, x_t, t)
        output = self.p_mean_variance(model, x_t, t, estimate_noise)
        kl = normal_kl(true_mean, true_log_var, output["mean"], output["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretised_gaussian_log_likelihood(
                x_0, output["mean"], log_scales=0.5 * output["log_variance"]
                )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        nll = torch.where((t == 0), decoder_nll, kl)
        return {"output": nll, "pred_x_0": output["pred_x_0"]}

    def calc_loss(self, model, x_0, t):
        # noise = torch.randn_like(x)

        noise = self.noise_fn(x_0, t).float()

        x_t = self.sample_q(x_0, t, noise)
        estimate_noise = model(x_t, t)
        loss = {}
        if self.loss_type == "l1":
            loss["loss"] = mean_flat((estimate_noise - noise).abs())
        elif self.loss_type == "l2":
            loss["loss"] = mean_flat((estimate_noise - noise).square())
        elif self.loss_type == "hybrid":
            # add vlb term
            loss["vlb"] = self.calc_vlb_xt(model, x_0, x_t, t, estimate_noise)["output"]
            loss["loss"] = loss["vlb"] + mean_flat((estimate_noise - noise).square())
        else:
            loss["loss"] = mean_flat((estimate_noise - noise).square())
        return loss, x_t, estimate_noise

    def p_loss(self, model, x_0, args):
        if self.loss_weight == "none":
            if args["train_start"]:
                t = torch.randint(
                        0, min(args["sample_distance"], self.num_timesteps), (x_0.shape[0],),
                        device=x_0.device
                        )
            else:
                t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
            weights = 1
        else:
            t, weights = self.sample_t_with_weights(x_0.shape[0], x_0.device)

        loss, x_t, eps_t = self.calc_loss(model, x_0, t)
        loss = ((loss["loss"] * weights).mean(), (loss, x_t, eps_t))
        return loss

    def prior_vlb(self, x_0, args):
        t = torch.tensor([self.num_timesteps - 1] * args["Batch_Size"], device=x_0.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
        kl_prior = normal_kl(
                mean1=qt_mean, logvar1=qt_log_variance, mean2=torch.tensor(0.0, device=x_0.device),
                logvar2=torch.tensor(0.0, device=x_0.device)
                )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_total_vlb(self, x_0, model, args):
        vb = []
        x_0_mse = []
        mse = []
        for t in reversed(list(range(self.num_timesteps))):
            t_batch = torch.tensor([t] * args["Batch_Size"], device=x_0.device)
            noise = torch.randn_like(x_0)
            x_t = self.sample_q(x_0=x_0, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self.calc_vlb_xt(
                        model,
                        x_0=x_0,
                        x_t=x_t,
                        t=t_batch,
                        )
            vb.append(out["output"])
            x_0_mse.append(mean_flat((out["pred_x_0"] - x_0) ** 2))
            eps = self.predict_eps_from_x_0(x_t, t_batch, out["pred_x_0"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = torch.stack(vb, dim=1)
        x_0_mse = torch.stack(x_0_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_vlb = self.prior_vlb(x_0, args)
        total_vlb = vb.sum(dim=1) + prior_vlb
        return {
            "total_vlb": total_vlb,
            "prior_vlb": prior_vlb,
            "vb":        vb,
            "x_0_mse":   x_0_mse,
            "mse":       mse,
            }

    def detection_A(self, model, x_0, args, file, mask, total_avg=2):
        for i in [f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}",
                  f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}/{file[1]}/",
                  f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}/{file[1]}/A"]:
            try:
                os.makedirs(i)
            except OSError:
                pass

        for i in range(7, 0, -1):
            freq = 2 ** i
            self.noise_fn = lambda x, t: generate_simplex_noise(
                    self.simplex, x, t, False, frequency=freq,
                    in_channels=self.img_channels
                    )

            for t_distance in range(50, int(args["T"] * 0.6), 50):
                output = torch.empty((total_avg, 1, *args["img_size"]), device=x_0.device)
                for avg in range(total_avg):

                    t_tensor = torch.tensor([t_distance], device=x_0.device).repeat(x_0.shape[0])
                    x = self.sample_q(
                            x_0, t_tensor,
                            self.noise_fn(x_0, t_tensor).float()
                            )

                    for t in range(int(t_distance) - 1, -1, -1):
                        t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                        with torch.no_grad():
                            out = self.sample_p(model, x, t_batch)
                            x = out["sample"]

                    output[avg, ...] = x

                # save image containing initial, each final denoised image, mean & mse
                output_mean = torch.mean(output, dim=0).reshape(1, 1, *args["img_size"])
                mse = ((output_mean - x_0).square() * 2) - 1
                mse_threshold = mse > 0
                mse_threshold = (mse_threshold.float() * 2) - 1
                out = torch.cat([x_0, output[:3], output_mean, mse, mse_threshold, mask])

                temp = os.listdir(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/A')

                plt.imshow(gridify_output(out, 4), cmap='gray')
                plt.axis('off')
                plt.savefig(
                        f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/A/freq={i}-t'
                        f'={t_distance}-{len(temp) + 1}.png'
                        )
                plt.clf()

    def detection_B(self, model, x_0, args, file, mask, denoise_fn="gauss", total_avg=5):
        assert type(file) == tuple
        for i in [f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}",
                  f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}/{file[1]}",
                  f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}/{file[1]}/{denoise_fn}"]:
            try:
                os.makedirs(i)
            except OSError:
                pass
        if denoise_fn == "octave":
            end = int(args["T"] * 0.6)
            self.noise_fn = lambda x, t: generate_simplex_noise(
                    self.simplex, x, t, False, frequency=64, octave=6,
                    persistence=0.8
                    ).float()
        else:
            end = int(args["T"] * 0.8)
            self.noise_fn = lambda x, t: torch.randn_like(x)
        # multiprocessing?
        dice_coeff = []
        for t_distance in range(50, end, 50):
            output = torch.empty((total_avg, 1, *args["img_size"]), device=x_0.device)
            for avg in range(total_avg):

                t_tensor = torch.tensor([t_distance], device=x_0.device).repeat(x_0.shape[0])
                x = self.sample_q(
                        x_0, t_tensor,
                        self.noise_fn(x_0, t_tensor).float()
                        )

                for t in range(int(t_distance) - 1, -1, -1):
                    t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                    with torch.no_grad():
                        out = self.sample_p(model, x, t_batch)
                        x = out["sample"]

                output[avg, ...] = x

            # save image containing initial, each final denoised image, mean & mse
            output_mean = torch.mean(output, dim=[0]).reshape(1, 1, *args["img_size"])

            temp = os.listdir(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/{denoise_fn}')

            dice = evaluation.heatmap(
                    real=x_0, recon=output_mean, mask=mask,
                    filename=f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/'
                             f'{denoise_fn}/heatmap-t={t_distance}-{len(temp) + 1}.png'
                    )

            mse = ((output_mean - x_0).square() * 2) - 1
            mse_threshold = mse > 0
            mse_threshold = (mse_threshold.float() * 2) - 1
            out = torch.cat([x_0, output[:3], output_mean, mse, mse_threshold, mask])

            plt.imshow(gridify_output(out, 4), cmap='gray')
            plt.axis('off')
            plt.savefig(
                    f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/{denoise_fn}/t'
                    f'={t_distance}-{len(temp) + 1}.png'
                    )
            plt.clf()

            dice_coeff.append(dice)
        return dice_coeff

    def detection_A_fixedT(self, model, x_0, args, mask, end_freq=6):
        t_distance = 250

        output = torch.empty((6 * end_freq, 1, *args["img_size"]), device=x_0.device)
        for i in range(1, end_freq + 1):

            freq = 2 ** i
            noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, False, frequency=freq).float()

            t_tensor = torch.tensor([t_distance - 1], device=x_0.device).repeat(x_0.shape[0])
            x = self.sample_q(
                    x_0, t_tensor,
                    noise_fn(x_0, t_tensor).float()
                    )
            x_noised = x.clone().detach()
            for t in range(int(t_distance) - 1, -1, -1):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                with torch.no_grad():
                    out = self.sample_p(model, x, t_batch, denoise_fn=noise_fn)
                    x = out["sample"]

            mse = ((x_0 - x).square() * 2) - 1
            mse_threshold = mse > 0
            mse_threshold = (mse_threshold.float() * 2) - 1

            output[(i - 1) * 6:i * 6, ...] = torch.cat((x_0, x_noised, x, mse, mse_threshold, mask))

        return output


x = """
Two methods of detection:

A - using varying simplex frequencies
B - using octave based simplex noise
C - gaussian based (same as B but gaussian)


A: for i in range(6,0,-1):
    2**i == frequency
   Frequency = 64: Sample 10 times at t=50, denoise and average
   Repeat at t = range (50, ARGS["sample distance"], 50)
   
   Note simplex noise is fixed frequency ie no octave mixure
   
B: Using some initial "good" simplex octave parameters such as 64 freq, oct = 6, persistence= 0.9   
   Sample 10 times at t=50, denoise and average
   Repeat at t = range (50, ARGS["sample distance"], 50)
   
"""
