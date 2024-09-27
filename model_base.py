
#!/usr/bin/env python
# coding: utf-8

# file: model_base.py
# -*- coding: utf-8 -*-

# importing the modules
import concurrent.futures

import numpy as np
import torch
import cv2
import gradio as gr
from PIL import Image
import huggingface_hub
import transformers
from transformers import pipeline
from transformers import CLIPTextModel
from transformers import AutoImageProcessor
from transformers import SegformerForSemanticSegmentation
from stable_diffusion_reference import StableDiffusionReferencePipeline
from controlnet_aux import HEDdetector
from controlnet_aux import LineartDetector
from controlnet_aux import OpenposeDetector
from controlnet_aux import LineartAnimeDetector
from diffusers import DiffusionPipeline
from diffusers import I2VGenXLPipeline
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionControlNetInpaintPipeline
from diffusers import StableDiffusionControlNetImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import load_image
from diffusers.utils import export_to_gif
from diffusers.utils import export_to_video
from compel import Compel
import onnxruntime as rt


# set gpu device
device = "cuda" if torch.cuda.is_available() else "cpu"

# options for input to print_msg function to decide what to print on screen
LOAD_MODEL_TEXT = "load model"
FINISH_MODEL_LOADING_TEXT = "finish model loading"

# some hidden text to enchance quality and reduce defects
hidden_booster_text = ", masterpiece-anatomy-perfect, dynamic, dynamic colors, bright colors, high contrast, excellent work, extremely elaborate picture description, 8k, obvious light and shadow effects, ray tracing, obvious layers, depth of field, best quality, RAW photo, best quality, highly detailed, intricate details, HD, 4k, 8k, high quality, beautiful eyes, sparkling eyes, beautiful face, masterpiece,best quality,ultimate details,highres,8k,wallpaper,extremely clear,"
hidden_negative = ", internal-organs-outside-the-body, internal-organs-visible, anatomy-description, unprompted-nsfw ,worst-human-external-anatomy, worst-human-hands-anatomy, worst-human-fingers-anatomy, worst-detailed-eyes, worst-detailed-fingers, worst-human-feet-anatomy, worst-human-toes-anatomy, worst-detailed-feet, worst-detailed-toes, camera, smartphone, worst-facial-details, ugly-detailed-fingers, ugly-detailed-toes,fingers-in-worst-possible-shape, worst-detailed-eyes, undetailed-eyes, undetailed-fingers, undetailed-toes, "


class Model():
    def __init__(self, model_name):
        self.model_name = model_name
        self.pipe = None
        
    def load_lora(self, pipe, chibi=False):
        pipe.load_lora_weights("shellypeng/lora1")
        pipe.fuse_lora(lora_scale=0.5)
        pipe.load_lora_weights("shellypeng/lora2")
        pipe.fuse_lora(lora_scale=0.6)
        if chibi:
            pipe.load_lora_weights("shellypeng/lora3")
            pipe.fuse_lora(lora_scale=2.0)
        
        return pipe
    def multi_thread_load_lora(self, pipe, chibi=False):
        # conduct multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
            future = executor.submit(self.load_lora, pipe, chibi)
            response = future.result()
        return response
    
    def load_ti(self, pipe):
        # load textual inversions
        pipe.load_textual_inversion("shellypeng/textinv1")
        pipe.load_textual_inversion("shellypeng/textinv2")
        pipe.load_textual_inversion("./EasyNegative.pt")
        pipe.load_textual_inversion("shellypeng/textinv3")
        pipe.load_textual_inversion("shellypeng/textinv4")
        
        return pipe
    
    def multi_thread_load_ti(self, pipe):
        # conduct multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
            future = executor.submit(self.load_ti, pipe)
            response = future.result()
        return response
    
    def load_model(self):
        gr.Info(LOAD_MODEL_TEXT)
        
        match self.model_name:
            case "chatbot":
                pipe = pipeline("text-generation", model="vicgalle/Roleplay-Llama-3-8B", device=device)

            case "txt2img":
                # load pipe
                pipe = DiffusionPipeline.from_pretrained(
                    "shellypeng/model_am", 
                    torch_dtype=torch.float16, 
                )
                # load LoRAs
                pipe = self.multi_thread_load_lora(pipe)
                # load textual inversions
                pipe = self.multi_thread_load_ti(pipe)

            case "scribble2img":
                # load ControlNet for scribble
                hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
                controlnet_scribble = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16, use_safetensors=True)
                self.hed = hed
                
                # load pipe
                pipe = StableDiffusionControlNetPipeline.from_pretrained("shellypeng/model_am", controlnet=controlnet_scribble, torch_dtype=torch.float16)
                # load HED processor for preprocessing of ControlNet scribble
                
                # load LoRAs
                pipe = self.multi_thread_load_lora(pipe)
                # load textual inversions
                pipe = self.multi_thread_load_ti(pipe)
                print("scribble loaded")
            case "img2img":
                # load pipe
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained("shellypeng/model_am", torch_dtype=torch.float16,)
                
                # load LoRAs
                pipe = self.multi_thread_load_lora(pipe)
                # load textual inversions
                pipe = self.multi_thread_load_ti(pipe)

            case "img2pose":
                # load pipe
                pipe = StableDiffusionReferencePipeline.from_pretrained(
                    "shellypeng/model_am", 
                    torch_dtype=torch.float16, 
                )
                # load LoRAs
                pipe = self.multi_thread_load_lora(pipe)
                # load textual inversions
                pipe = self.multi_thread_load_ti(pipe)

            case "inpaint":
                # load ControlNet for inpainting
                controlnet_inpaint = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16")

                # load pipe
                pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    "shellypeng/model_am", controlnet=controlnet_inpaint, torch_dtype=torch.float16
                )
                
                
                # load LoRAs
                pipe = self.multi_thread_load_lora(pipe)
                # load textual inversions
                pipe = self.multi_thread_load_ti(pipe)

            
            case "chibi":
                # load pipe
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained("shellypeng/model_am",
                                                                            torch_dtype=torch.float16, )
                # load LoRAs
                pipe = self.multi_thread_load_lora(pipe, chibi=True)
                # load textual inversions
                pipe = self.multi_thread_load_ti(pipe)

            case "lineart2img":
                # load lineart processor for preprocessing of the ControlNet
                lineart_processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
                controlnet_lineart = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15s2_lineart_anime", torch_dtype=torch.float16, )
                self.lineart_processor = lineart_processor
                
                # load pipe
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    "shellypeng/model_am", controlnet=controlnet_lineart,
                    torch_dtype=torch.float16, 
                )
                
                # load LoRAs
                pipe = self.multi_thread_load_lora(pipe)
                # load textual inversions
                pipe = self.multi_thread_load_ti(pipe)

            
            case "img2video":
                # load pipe
                pipe = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16)
                
            case "rmbg":
                # load pipe
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                model_path = huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
                pipe = rt.InferenceSession(model_path, providers=providers)
                
        # load scheduler
        if self.model_name != "chatbot" and self.model_name != "img2video" and self.model_name != "rmbg":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
            
        # send to gpu
        if self.model_name != "chatbot" and self.model_name != "rmbg":
            pipe.to(device)
        
        gr.Info(FINISH_MODEL_LOADING_TEXT)
        self.pipe = pipe
        return pipe
        
    def multi_thread_load_model(self):
        # conduct multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
            future = executor.submit(self.load_model)
            response = future.result()
        return response
    
    def check_input_img(self, image):
        if image is None:
            raise gr.Error("Please provide a input image.")
        
    def check_height_width(self, height, width):
        if height % 8:
            raise gr.Error("Please input a height with a value of multiple of 8 on the slider.")
            
        if width % 8:
            raise gr.Error("Please input a width with a value of multiple of 8 on the slider.")
        
    def compel_prompts(self, prompt, negative_prompt):
        # prompt weighter to add weights to prompts
        compel_proc = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder)
  
        # positive prompt with weights
        prompt = prompt + hidden_booster_text 

        prompt_embeds = compel_proc(prompt)

        # negative prompt with weights
        negative_prompt = negative_prompt + hidden_negative
        negative_prompt_embeds = compel_proc(negative_prompt)
        
        return prompt_embeds, negative_prompt_embeds

    def gen_batch_img(self, pipe, height, width, prompt_embeds, negative_prompt_embeds, num_inference_steps, input_img=None, strength=1.0, control_image=None, mask_image=None, num_images=4):
        res = []
        for x in range(num_images):
            i = 0
            res_image = Image.fromarray(np.zeros((64, 64)))
            while not res_image.getbbox() and i < 30:
                if isinstance(pipe, StableDiffusionReferencePipeline):
                    res_image = pipe(ref_image=input_img,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        num_inference_steps=num_inference_steps,
                        reference_attn=True,
                        reference_adain=True).images[0]
                elif isinstance(pipe, StableDiffusionControlNetInpaintPipeline):
                    res_image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=input_img, strength=strength, num_inference_steps=num_inference_steps, control_image=control_image, mask_image=mask_image).images[0]
                elif isinstance(pipe, StableDiffusionImg2ImgPipeline):
                    res_image = pipe(height=height, width=width, image=input_img, strength=0.6, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_inference_steps=40).images[0]
                elif input_img is not None:
                    res_image = pipe(height=height, width=width, image=input_img, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_inference_steps=40).images[0]
                else:
                    res_image = pipe(height=height, width=width, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_inference_steps=num_inference_steps).images[0]
                i += 1
            res.append(res_image)
        return res
    
    def delete_model(self):
        # delete models to release memory
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        else:
            raise gr.Error("No model to delete.")
            
        # print message when finished deleting models
        gr.Info("Model deleted.")
    
class ChatbotModel(Model):
    def __init__(self):
        super().__init__("chatbot")
        
    def infer(self, **kwargs):
        response = self.pipe(kwargs["messages"], max_new_tokens=kwargs["max_new_tokens"], do_sample=kwargs["do_sample"])[0]["generated_text"][1]["content"]
        return response
            
class Txt2imgModel(Model):
    def __init__(self):
        super().__init__("txt2img")
        
    def infer(self, **kwargs):
        # get argument names from kwargs for convenience
        prompt, negative_prompt, height, width, num_images = kwargs["prompt"], kwargs["negative_prompt"], kwargs["height"], kwargs["width"], kwargs["num_images"]
        
        # check valid height and width
        self.check_height_width(height, width)
        
        # compel(i.e. add control to) the prompts and negative prompts
        prompt_embeds, negative_prompt_embeds = self.compel_prompts(prompt, negative_prompt)
        
        # generate result image(s)
        res = self.gen_batch_img(self.pipe, height, width, prompt_embeds, negative_prompt_embeds, 40, num_images=num_images)

        return res
    
    
class Scribble2imgModel(Model):
    def __init__(self):
        super().__init__("scribble2img")
        
    def infer(self, **kwargs):
        # get argument names from kwargs for convenience
        prompt, negative_prompt, input_img = kwargs["prompt"], kwargs["negative_prompt"], kwargs["input_img"]
        
        # check valid input image
        self.check_input_img(input_img)
        
        width, height = input_img.size
        
        # preprocessing input image
        input_img = np.array(input_img)
        input_img[input_img > 100] = 255
        input_img = Image.fromarray(input_img)
        input_img = self.hed(input_img, scribble=True)
        input_img.save("hed_img.png")
        input_img = load_image(input_img)
        
        # compel(i.e. add control to) the prompts and negative prompts
        prompt_embeds, negative_prompt_embeds = self.compel_prompts(prompt, negative_prompt)

        # generating 4 result images
        res = self.gen_batch_img(self.pipe, height, width, prompt_embeds, negative_prompt_embeds, 40, input_img, 4)
        
        return res
        
class Img2imgModel(Model):
    def __init__(self):
        super().__init__("img2img")
        
    def infer(self, **kwargs):
        # get argument names from kwargs for convenience
        prompt, negative_prompt, input_img = kwargs["prompt"], kwargs["negative_prompt"], kwargs["input_img"]
        
        # check valid input image
        self.check_input_img(input_img)
        
        # get input image
        input_img = load_image(input_img)
        width, height = input_img.size
        
        # compel(i.e. add control to) the prompts and negative prompts
        prompt_embeds, negative_prompt_embeds = self.compel_prompts(prompt, negative_prompt)

        # generating 4 result images
        res = self.gen_batch_img(self.pipe, height, width, prompt_embeds, negative_prompt_embeds, 40, input_img, 4)

        return res
        
        
class Img2poseModel(Model):
    def __init__(self):
        super().__init__("img2pose")
        
    def infer(self, **kwargs):
        # get argument names from kwargs for convenience
        prompt, negative_prompt, input_img = kwargs["prompt"], kwargs["negative_prompt"], kwargs["input_img"]
        
        # check valid input image
        self.check_input_img(input_img)
        
        # load image
        image = load_image(input_img)
        height, width = image.size

        # compel(i.e. add control to) the prompts and negative prompts
        prompt_embeds, negative_prompt_embeds = self.compel_prompts(prompt, negative_prompt)

        # generating 4 result images
        res = self.gen_batch_img(self.pipe, height, width, prompt_embeds, negative_prompt_embeds, 40, image, 4)

        return res
        
        
class InpaintModel(Model):
    def __init__(self):
        super().__init__("inpaint")
        
    # preprocess for inpainting
    def make_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image
    
    # fills the inpainted outline
    def fill_mask(self, mask_image):
        # convert mask to numpy array
        mask_image = np.array(mask_image)
        
        # obtain contours of enclosed shapes
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # fill the enclosed contours with white color
        cv2.drawContours(mask_image, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
        mask_image = Image.fromarray(mask_image)
        
        return mask_image
        
    def infer(self, **kwargs):
        # get argument names from kwargs for convenience
        prompt, negative_prompt, input_img = kwargs["prompt"], kwargs["negative_prompt"], kwargs["input_img"]
        
        # check valid input image
        self.check_input_img(input_img)
        
        # compel(i.e. add control to) the prompts and negative prompts
        prompt_embeds, negative_prompt_embeds = self.compel_prompts(prompt, negative_prompt)
        
        # set strength - how dissimilar to the reference image of the generated images
        strength = 0.9
        
        # load the mask layer of input image
        mask_image = load_image(input_img["layers"][0])

        # convert to numpy for preprocessing
        mask_image = np.array(mask_image)
        
        # convert user's mask outline to white for ControlNet Inpaint Pipeline to recognize(it can only recognize monochrome images)
        mask_image[np.all(mask_image == [93, 63, 211], axis=-1)] = [255, 255, 255]
        mask_image = Image.fromarray(mask_image).convert('L')

        # fill the enclosed outline the user has drawn
        mask_image = self.fill_mask(mask_image)
        image = load_image(input_img["background"])
        image_shape = image.size
        mask_image.resize(image_shape)
        height, width = image_shape

        # make the control image
        control_image = self.make_inpaint_condition(image, mask_image)

        # generating 4 result images
        res = self.gen_batch_img(self.pipe, height, width, prompt_embeds, negative_prompt_embeds, 40, image, strength, control_image, mask_image, 4)

        return res
        
        
class ChibiModel(Model):
    def __init__(self):
        super().__init__("chibi")
        
    def infer(self, **kwargs):
        # get argument names from kwargs for convenience
        prompt, negative_prompt, input_img = kwargs["prompt"], kwargs["negative_prompt"], kwargs["input_img"]
        
        # check valid input image
        self.check_input_img(input_img)
        
        # get height, width of input image
        width, height = input_img.size

        # compel(i.e. add control to) the prompts and negative prompts
        prompt_embeds, negative_prompt_embeds = self.compel_prompts("chibi+++" + prompt, negative_prompt)
        
        # preprocessing input image
        image = load_image(input_img)

        # generating 4 result images
        res = self.gen_batch_img(self.pipe, height, width, prompt_embeds, negative_prompt_embeds, 40, image, strength=0.5, num_images=4)

        return res
        
        
class Lineart2imgModel(Model):
    def __init__(self):
        super().__init__("lineart2img")
        
    def infer(self, **kwargs):
        # get argument names from kwargs for convenience
        prompt, negative_prompt, input_img = kwargs["prompt"], kwargs["negative_prompt"], kwargs["input_img"]
        
        # check valid input image
        self.check_input_img(input_img)
        
        # compel(i.e. add control to) the prompts and negative prompts
        prompt_embeds, negative_prompt_embeds = self.compel_prompts(prompt, negative_prompt)
        
        # preprocessing input image
        lineart_image = load_image(input_img)
        width, height = lineart_image.size
        lineart_image = self.lineart_processor(lineart_image)
        
        # generating 4 result images
        res = self.gen_batch_img(self.pipe, height, width, prompt_embeds, negative_prompt_embeds, 40, lineart_image, num_images=4)
        return res
        
        
class Img2videoModel(Model):
    def __init__(self):
        super().__init__("img2video")
        
    def infer(self, **kwargs):
        # get argument names from kwargs for convenience
        prompt, negative_prompt, num_inference_steps, num_frames, input_image, fps = kwargs["prompt"], kwargs["negative_prompt"], kwargs["num_inference_steps"], kwargs["num_frames"], kwargs["input_img"], kwargs["fps"]
        
        # check valid input image
        self.check_input_img(input_image)
        
        # get attributes - height, size, width - from input image
        width, height = input_image.size
        
        # generate result video
        output = self.pipe(
            prompt=prompt+hidden_booster_text,
            negative_prompt=negative_prompt+hidden_negative,
            image=input_image,
            target_fps=fps,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width
        ).frames[0]
        
        # save video for preview on Gradio app interface
        export_to_video(output, "I2VGen_video1.mp4")
        out_file_name = "I2VGen_video1.mp4"
        
        return out_file_name
    
class RMBGModel(Model):
    def __init__(self):
        super().__init__("rmbg")
    
    # get a mask for background removal
    def get_mask(self, img, s=1024):
        img = (img / 255).astype(np.float32)
        h, w = h0, w0 = img.shape[:-1]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        img_input = np.zeros([s, s, 3], dtype=np.float32)
        img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        mask = self.pipe.run(None, {'img': img_input})[0][0]
        mask = np.transpose(mask, (1, 2, 0))
        mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
        return mask
        
    def infer(self, **kwargs):
        # get argument names from kwargs for convenience
        input_img = kwargs["input_img"]
        
        # check valid input image
        self.check_input_img(input_img)
        
        # remove background
        img = np.array(input_img)
        mask = self.get_mask(img)
        img = (mask * img + 255 * (1 - mask)).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)
        img = np.concatenate([img, mask], axis=2, dtype=np.uint8)
        mask = mask.repeat(3, axis=2)
        img = Image.fromarray(img)
        return img
    
                
        
