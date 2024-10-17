"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps # 1000
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        # 50, 0.0, False
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        """
        array([  1,  21,  41,  61,  81, 101, 121, 141, 161, 181, 201, 221, 241,
                261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501,
                521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761,
                781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981])
        """
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,                      # n_samples, 4
               shape,                           # [4, 32, 32]z
               conditioning=None,               # 들어온다! c, (bs, 77, 1280)
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,                          # 0.0
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,                    # False
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1., # 5.0
               unconditional_conditioning=None, # 들어온다! uc, (bs, 77, 1280)
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:        # c로 제공됨.
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
                          # 50, 0.0, False
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning,                       # c, (bs, 77, 1280)
                                                    size,                               # (n_samples, C, H, W) (4, 4, 32, 32)
                                                    callback=callback,                  # None
                                                    img_callback=img_callback,          # None
                                                    quantize_denoised=quantize_x0,      # False
                                                    mask=mask, x0=x0,                   # 둘 다 None
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,        # 0.
                                                    temperature=temperature,            # 1.
                                                    score_corrector=score_corrector,    # None
                                                    corrector_kwargs=corrector_kwargs,  # None
                                                    x_T=x_T,                            # None
                                                    log_every_t=log_every_t,            # 100
                                                    unconditional_guidance_scale=unconditional_guidance_scale,  # 5.0
                                                    unconditional_conditioning=unconditional_conditioning,      # uc (bs, 77, 1280)
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, 
                      cond,             # c, (bs, 77, 1280)
                      shape,            # (n_samples, C, H, W) (4, 4, 32, 32)
                      x_T=None, 
                      ddim_use_original_steps=False,
                      callback=None, 
                      timesteps=None, 
                      quantize_denoised=False,
                      mask=None, 
                      x0=None, 
                      img_callback=None, 
                      log_every_t=100,
                      temperature=1., 
                      noise_dropout=0., 
                      score_corrector=None, 
                      corrector_kwargs=None,
                      unconditional_guidance_scale=1.,      # 5.0
                      unconditional_conditioning=None,):    # uc, (bs, 77, 1280)
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            # x_T가 없다면, 최초 x_T는 random noise!!!!
            img = torch.randn(shape, device=device) # (4, 4, 32, 32)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        """
        timesteps
        array([  1,  21,  41,  61,  81, 101, 121, 141, 161, 181, 201, 221, 241,
               261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501,
               521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761,
               781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981])
        """
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        """
        time_range
        array([981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
               721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481,
               461, 441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221,
               201, 181, 161, 141, 121, 101,  81,  61,  41,  21,   1])
        """
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]  # 50        
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        
        for i, step in enumerate(iterator):
            index = total_steps - i - 1 # 48 ~ 0
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            # ts: tensor([961, 961, 961, 961]) ~~> tensor([1, 1, 1, 1])

            if mask is not None: # mask==None이라 skip
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img,          # random noise ~> 생성된 이미지
                                      cond,         # c 
                                      ts,           # 
                                      index=index,  # 48 ~> 0
                                      use_original_steps=ddim_use_original_steps,   # False
                                      quantize_denoised=quantize_denoised,          # False
                                      temperature=temperature,                      # 1.
                                      noise_dropout=noise_dropout,                  # 0.
                                      score_corrector=score_corrector,              # None
                                      corrector_kwargs=corrector_kwargs,            # None
                                      unconditional_guidance_scale=unconditional_guidance_scale,    # 5.0
                                      unconditional_conditioning=unconditional_conditioning)        # uc, (bs, 77, 1280)
            
            img, pred_x0 = outs # step 돌며 이미지 생성
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, 
                      x,                    # img, (4, 4, 32, 32)
                      c,                    # c, (bs, 77, 1280)
                      t,                    # n_sample 길이의 1d array, 값은 961 ~ 1까지 변함.
                      index,                # 48 ~ 0 까지 변함
                      repeat_noise=False,   #
                      use_original_steps=False, 
                      quantize_denoised=False,
                      temperature=1., 
                      noise_dropout=0., 
                      score_corrector=None, 
                      corrector_kwargs=None,
                      unconditional_guidance_scale=1.,  # 5.0
                      unconditional_conditioning=None): # uc, (bs, 77, 1280)
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            # 여기로 들어온다.
            x_in = torch.cat([x] * 2) #uc와 c에 대해 함께 생성하기 위함으로 보여짐.
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])   # c_in의 shape는 (n_samples*2, max_seq_len, n_embed), ex에서는 (8, 77, 1280)
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2) # chunk(2)를 하는 이유는, uc와 c를 합쳐뒀기 때문이다
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None: # 안함
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised: # 안해
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
