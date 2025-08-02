import os, json, requests, random, time, cv2, ffmpeg, runpod
from urllib.parse import urlsplit

import torch
import numpy as np
from PIL import Image

from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_wan, nodes_sd3, nodes_model_advanced

CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
CLIPVisionLoader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
EmptySD3LatentImage = nodes_sd3.NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
ConditioningZeroOut = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()

LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
CLIPVisionEncode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
WanImageToVideo = nodes_wan.NODE_CLASS_MAPPINGS["WanImageToVideo"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
ModelSamplingSD3 = nodes_model_advanced.NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

with torch.inference_mode():
    flux_unet = UNETLoader.load_unet("FLUX-KREA-BLAZE-v1.safetensors", "fp8_e4m3fn_fast")[0]
    flux_clip = DualCLIPLoader.load_clip("clip_l.safetensors", "t5xxl_fp16.safetensors", "flux")[0]
    flux_vae = VAELoader.load_vae("ae.safetensors")[0]
    unet, clip, vae = CheckpointLoaderSimple.load_checkpoint("wan2.2-i2v-rapid-aio.safetensors")
    clip_vision = CLIPVisionLoader.load_clip("clip_vision_vit_h.safetensors")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

def images_to_mp4(images, output_path, fps=24):
    try:
        frames = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            if img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            frames.append(img)
        temp_files = [f"temp_{i:04d}.png" for i in range(len(frames))]
        for i, frame in enumerate(frames):
            success = cv2.imwrite(temp_files[i], frame[:, :, ::-1])
            if not success:
                raise ValueError(f"Failed to write {temp_files[i]}")
        if not os.path.exists(temp_files[0]):
            raise FileNotFoundError("Temporary PNG files were not created")
        stream = ffmpeg.input('temp_%04d.png', framerate=fps)
        stream = ffmpeg.output(stream, output_path, vcodec='libx264', pix_fmt='yuv420p')
        ffmpeg.run(stream, overwrite_output=True)
        for temp_file in temp_files:
            os.remove(temp_file)
    except Exception as e:
        print(f"Error: {e}")

@torch.inference_mode()
def generate_wan(input):
    try:
        values = input["input"]

        positive_prompt = values['positive_prompt'] # Fashion magazine, dynamic blur, hand-held lens, a close-up photo, the scene of a group of 21-year-old goths at a warehouse party, with a movie-like texture, super-realistic effect, realism.
        negative_prompt = values['negative_prompt'] # 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走
        crop = values['crop'] # center
        width = values['width'] # 1280
        height = values['height'] # 530
        length = values['length'] # 53
        batch_size = values['batch_size'] # 1
        shift = values['shift'] # 8.0
        cfg = values['cfg'] # 1.0
        sampler_name = values['sampler_name'] # lcm
        scheduler = values['scheduler'] # beta
        flux_sampler_name = values['flux_sampler_name'] # dpmpp_sde_gpu
        flux_scheduler = values['flux_scheduler'] # beta
        steps = values['steps'] # 4
        seed = values['seed'] # 0
        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)
        fps = values['fps'] # 24        
        
        flux_empty_latent = EmptySD3LatentImage.generate(width, height, batch_size)[0]
        flux_positive = CLIPTextEncode.encode(flux_clip, positive_prompt)[0]
        flux_negative = ConditioningZeroOut.zero_out(flux_positive)[0]
        flux_out_samples = KSampler.sample(flux_unet, seed, steps, cfg, flux_sampler_name, flux_scheduler, flux_positive, flux_negative, flux_empty_latent)[0]
        flux_decoded_images = VAEDecode.decode(flux_vae, flux_out_samples)[0].detach()
        flux_image = Image.fromarray(np.array(flux_decoded_images*255, dtype=np.uint8)[0]).save(f"/content/flux_image.png")
    
        input_image = f"/content/flux_image.png"

        model = ModelSamplingSD3.patch(unet, shift)[0]
        positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode.encode(clip, negative_prompt)[0]

        input_image = LoadImage.load_image(input_image)[0]
        clip_vision_output = CLIPVisionEncode.encode(clip_vision, input_image, crop)[0]
        positive, negative, out_latent = WanImageToVideo.encode(positive, negative, vae, width, height, length, batch_size, start_image=input_image, clip_vision_output=clip_vision_output)
        out_samples = KSampler.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, out_latent)[0]

        decoded_images = VAEDecode.decode(vae, out_samples)[0].detach()
        images_to_mp4(decoded_images, f"/content/flux-krea-blaze-wan2.2-i2v-rapid-{seed}-tost.mp4", fps)
        
        result = f"/content/flux-krea-blaze-wan2.2-i2v-rapid-{seed}-tost.mp4"

        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        with open(result, 'rb') as file:
            response = requests.post("https://upload.tost.ai/api/v1", files={'file': file})
        response.raise_for_status()
        result_url = response.text
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})