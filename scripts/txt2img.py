import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    # breakpoint()
    model = instantiate_from_config(config.model) # LatentDiffusion!!!
    m, u = model.load_state_dict(sd, strict=False) # satedict로 불러오기
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()


    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    breakpoint()
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path
    breakpoint()
    # model: LatentDiffusion(DDPM)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model) # 아마? 이걸로 선택될 것

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0: # opt.scale == 5.0
                uc = model.get_learned_conditioning(opt.n_samples * [""])       # model.cond_stage_model.encode 를 수행한다. cond_stage_model은 BERTEmbedder이고, BERTEmbedder의 .encode함수는 입력 opt.n_samples * [""]를 입력으로 해 forward 진행
                                                                                # text token화 후, TrasformerWrapper를 통과해 (bs, 77, 1280)으로 임베딩! 하지만 uc는 아무것도 들어있지 않다..
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])    # 위 uc와 동일하게 진행되나, prompt의 Encoding을 진행함에 차이가 있다.
                shape = [4, opt.H//8, opt.W//8] # default: [4, 32, 32]
                # 이놈의 sampler가 쌈뽕짝한 것들을 담당할 것 같다... 하지만 DDIMSampler안에 model, 즉 LatentDiffusion이 parameter로 제공되기 때문에,
                # 우선은 LatentDiffusion의 코드 진행에 대해 파악하자. 지금까지 AutoencodkerKL, BERTEmbedder 完. 밥을 먹고 와서 UNet을 파악 하자!!!!!! 젭라!!~!~!~
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps, # 50              
                                                 conditioning=c,    # c는 (bs, 77, 1280)의 shape을 가진 Tensor일 것이다.
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)     # 최종적으로 나온 sample의 ddim을 제공, model.first_stage_model.decode(samples_ddim)이 진행됨. (samples_ddim 내부에서 조금 수정됨)
                                                                            # model.first_stage_model은 AutoencoderKL이고, AutoencoderKL의 decode함수는 AutoencoderKL.post_quant_conv(z) -> AutoencoderKL.decoder(z) 통과해 return.
                                                                            # post_quant_conv는 Convolution으로, channel에 맞게 복원하는 과정이라..고 생각한다.
                                                                            # AutoencoderKL.decoder는 module Decoder, Upsampling을 진행하는 Decoder이다.

                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                
                # sample이 4개 나왔따~!
                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                    base_count += 1
                all_samples.append(x_samples_ddim)


    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
