import torch
import os
import tempfile
import time
import random
import gradio as gr

gradio_temp_dir = os.path.join('output')
os.makedirs(gradio_temp_dir, exist_ok=True)


from PIL import Image
from diffusers import (
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,  # <-- Added import
    EulerDiscreteScheduler  # <-- Added import
)

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# Initialize both pipelines
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)

# Disable the safety checker
SAFETY_CHECKER_ENABLED = os.environ.get("SAFETY_CHECKER", "0") == "1"
safety_checker = None
feature_extractor = None

main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    vae=vae,
    safety_checker=safety_checker,
    feature_extractor=feature_extractor,
    torch_dtype=torch.float16,
).to(device)

image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)

# Sampler map
SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}

def center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size

    # Calculate dimensions to crop to the center
    new_dimension = min(width, height)
    left = (width - new_dimension)/2
    top = (height - new_dimension)/2
    right = (width + new_dimension)/2
    bottom = (height + new_dimension)/2

    # Crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)

    return img

def common_upscale(samples, width, height, upscale_method, crop=False):
        if crop == "center":
            old_width = samples.shape[3]
            old_height = samples.shape[2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples[:,:,y:old_height-y,x:old_width-x]
        else:
            s = samples

        return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

def upscale(samples, upscale_method, scale_by):
        #s = samples.copy()
        width = round(samples["images"].shape[3] * scale_by)
        height = round(samples["images"].shape[2] * scale_by)
        s = common_upscale(samples["images"], width, height, upscale_method, "disabled")
        return (s)

def check_inputs(prompt: str, control_image: Image.Image):
    if control_image is None:
        raise gr.Error("Please select or upload an Input Illusion")
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")

def convert_to_pil(base64_image):
    pil_image = Image.open(base64_image)
    return pil_image

def convert_to_base64(pil_image):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        image.save(temp_file.name)
    return temp_file.name

# Inference function

def inference(
    control_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 8.0,
    controlnet_conditioning_scale: float = 1,
    control_guidance_start: float = 1,    
    control_guidance_end: float = 1,
    upscaler_strength: float = 0.5,
    seed: int = -1,
    sampler = "DPM++ Karras SDE",
    progress = gr.Progress(track_tqdm=True),
    profile: gr.OAuthProfile | None = None,
):
    start_time = time.time()
    start_time_struct = time.localtime(start_time)
    start_time_formatted = time.strftime("%H:%M:%S", start_time_struct)
    print(f"Inference started at {start_time_formatted}")
    
    # Rest of your existing code
    control_image_small = center_crop_resize(control_image)
    control_image_large = center_crop_resize(control_image, (1024, 1024))

    main_pipe.scheduler = SAMPLER_MAP[sampler](main_pipe.scheduler.config)
    my_seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    generator = torch.Generator(device=device).manual_seed(my_seed)
    
    out = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image_small,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        num_inference_steps=15,
        output_type="latent"
    )
    upscaled_latents = upscale(out, "nearest-exact", 2)
    out_image = image_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=control_image_large,        
        image=upscaled_latents,
        guidance_scale=float(guidance_scale),
        generator=generator,
        num_inference_steps=20,
        strength=upscaler_strength,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale)
    )
    end_time = time.time()
    end_time_struct = time.localtime(end_time)
    end_time_formatted = time.strftime("%H:%M:%S", end_time_struct)
    print(f"Inference ended at {end_time_formatted}, taking {end_time-start_time}s")


    # Save the output image to the gradio_temp_dir
    image_filename = f"output_{start_time}.png"
    image_filepath = os.path.join(gradio_temp_dir, image_filename)
    out_image["images"][0].save(image_filepath)
    print(f"Image saved to: {image_filepath}")

    return out_image["images"][0], gr.update(visible=True), gr.update(visible=True), my_seed
    
        
        
with gr.Blocks() as app:
    gr.Markdown(
        '''
        <div style="text-align: center;">
            <h1>Illusion Diffusion HQ 🌀</h1>
            <p style="font-size:16px;">Generate stunning high quality illusion artwork with Stable Diffusion</p>
            <p>Illusion Diffusion is back up with a safety checker! Because I have been asked, if you would like to support me, consider using <a href="https://deforum.studio">deforum.studio</a></p>
            <p>A space by AP <a href="https://twitter.com/angrypenguinPNG">Follow me on Twitter</a> with big contributions from <a href="https://twitter.com/multimodalart">multimodalart</a></p>
            <p>This project works by using <a href="https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster">Monster Labs QR Control Net</a>. Given a prompt and your pattern, we use a QR code conditioned controlnet to create a stunning illusion! Credit to: <a href="https://twitter.com/MrUgleh">MrUgleh</a> for discovering the workflow :)</p>
        </div>
        '''
    )


    state_img_input = gr.State()
    state_img_output = gr.State()
    with gr.Row():
        with gr.Column():
            control_image = gr.Image(label="Input Illusion", type="pil", elem_id="control_image")
            controlnet_conditioning_scale = gr.Slider(minimum=0.0, maximum=5.0, step=0.01, value=0.65, label="Illusion strength", elem_id="illusion_strength", info="ControlNet conditioning scale")
            gr.Examples(examples=["examples/checkers.png", "examples/pattern.png", "examples/checkers_mid.jpg", "examples/ultra_checkers.png", "examples/cubed.png", "examples/spiral.jpeg", "examples/funky.jpeg"], inputs=control_image)
            prompt = gr.Textbox(label="Prompt", elem_id="prompt", info="Type what you want to generate", placeholder="Medieval village scene with busy streets and castle in the distance")
            negative_prompt = gr.Textbox(label="Negative Prompt", info="Type what you don't want to see", value="low quality", elem_id="negative_prompt")
            with gr.Accordion(label="Advanced Options", open=False):
                guidance_scale = gr.Slider(minimum=0.0, maximum=50.0, step=0.25, value=7.5, label="Guidance Scale")
                sampler = gr.Dropdown(choices=list(SAMPLER_MAP.keys()), value="Euler")
                control_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0, label="Start of ControlNet")
                control_end = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=1, label="End of ControlNet")
                strength = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=1, label="Strength of the upscaler")
                seed = gr.Slider(minimum=-1, maximum=9999999999, step=1, value=-1, label="Seed", info="-1 means random seed")
                used_seed = gr.Number(label="Last seed used",interactive=False)
            run_btn = gr.Button("Run")
        with gr.Column():
            result_image = gr.Image(label="Illusion Diffusion Output", interactive=False, elem_id="output")
            
    prompt.submit(
        check_inputs,
        inputs=[prompt, control_image],
        queue=False
    ).success(
        inference,
        inputs=[control_image, prompt, negative_prompt, guidance_scale, controlnet_conditioning_scale, control_start, control_end, strength, seed, sampler],
        outputs=[result_image, used_seed])
    
    run_btn.click(
        check_inputs,
        inputs=[prompt, control_image],
        queue=False
    ).success(
        inference,
        inputs=[control_image, prompt, negative_prompt, guidance_scale, controlnet_conditioning_scale, control_start, control_end, strength, seed, sampler],
        outputs=[result_image, used_seed])
        
app.queue(max_size=20)

if __name__ == "__main__":
    app.launch()