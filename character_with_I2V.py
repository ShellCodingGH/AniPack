#!/usr/bin/env python
# coding: utf-8

# file: character_with_I2V.py
# -*- coding: utf-8 -*-

# import the modules
HF_TOKEN = "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"
import torch
import numpy as np
import cv2
import os
from moviepy.editor import *
import torchvision
from torchvision.io import write_video
import moviepy.editor as mp
from random import randrange
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from stable_diffusion_reference import StableDiffusionReferencePipeline
import gradio as gr
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionInpaintPipeline
from controlnet_aux import HEDdetector, LineartDetector, OpenposeDetector, LineartAnimeDetector
from diffusers.utils import load_image
from diffusers import DiffusionPipeline
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image
import concurrent.futures
from compel import Compel
import onnxruntime as rt
import imageio
import torchvision.transforms as transforms
import transformers
from transformers import pipeline, CLIPTextModel
from huggingface_hub import login
from PIL import ImageDraw
import huggingface_hub
import argparse
import json
import os
import re
import tempfile
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
import librosa
import numpy as np
import torch
from torch import no_grad, LongTensor
import commons
import utils
import gradio as gr
import gradio.utils as gr_utils
import gradio.processing_utils as gr_processing_utils
import ONNXVITS_infer
import models
from text import text_to_sequence, _clean_text
from text.symbols import symbols
from mel_processing import spectrogram_torch
import psutil
from datetime import datetime

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}

# limit text and audio length in huggingface spaces
limitation = os.getenv("SYSTEM") == "spaces"  

# set gpu device
device = "cuda"

# some hidden text to enchance quality and reduce defects
hidden_booster_text = ", masterpiece-anatomy-perfect, dynamic, dynamic colors, bright colors, high contrast, excellent work, extremely elaborate picture description, 8k, obvious light and shadow effects, ray tracing, obvious layers, depth of field, best quality, RAW photo, best quality, highly detailed, intricate details, HD, 4k, 8k, high quality, beautiful eyes, sparkling eyes, beautiful face, masterpiece,best quality,ultimate details,highres,8k,wallpaper,extremely clear,"
hidden_negative = ", boobs++, internal-organs-outside-the-body, internal-organs-visible, anatomy-description, unprompted-nsfw ,worst-human-external-anatomy, worst-human-hands-anatomy, worst-human-fingers-anatomy, worst-detailed-eyes, worst-detailed-fingers, worst-human-feet-anatomy, worst-human-toes-anatomy, worst-detailed-feet, worst-detailed-toes, camera, smartphone, worst-facial-details, ugly-detailed-fingers, ugly-detailed-toes,fingers-in-worst-possible-shape, worst-detailed-eyes, undetailed-eyes, undetailed-fingers, undetailed-toes, boobs, big boobs, huge boobs, sexy, dirty, d cup, e cup, g cup, slutty, bad-image-v2-39000,badhandv4,ng_deepnegative_v1_75t,EasyNegative,bad_prompt_version2, worst quality, low quality, extra digits, text, signature, bad anatomy, mutated hand, error, missing finger, cropped, worse quality, bad quality, lowres, floating limbs, bad hands, anatomical nonsense, Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
hidden_vary_negative = ""

# login to huggingface for use of models
login(HF_TOKEN)

# list of congratulation voice messages
audio_list = ["congrats-audios/audio1.wav", "congrats-audios/audio2.wav", "congrats-audios/audio3.wav", "congrats-audios/audio4.wav", 
             "congrats-audios/audio5.wav", "congrats-audios/audio6.wav", "congrats-audios/audio7.wav", "congrats-audios/audio8.wav", 
             "congrats-audios/audio9.wav", "congrats-audios/audio10.wav", "congrats-audios/audio11.wav", "congrats-audios/audio12.wav", 
             ]

# plays congratulation voice messages
def play_audio():
    audio = audio_list[randrange(len(audio_list))]
    return audio

def mult_thread_audio():
    
    # conduct multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        future = executor.submit(play_audio)
        audio = future.result()
    return audio

# print some messages on screen when certain features are triggered
def print_msg(option):
    if option == "load model":
        gr.Info("Loading model")
    elif option == "finish model loading":
        gr.Info("Finished loading model")

# options for input to print_msg function to decide what to print on screen
LOAD_MODEL_OPTION = "load model"
FINISH_MODEL_LOADING_OPTION = "finish model loading"

# number of models used in this app
num_models = 10
pipe_txt2img, pipe_img2img, pipe_scribble, pipe_lineart, pipe_pose, pipe_video, pipe_chibi, pipe_inpaint, pipe_chatbot, rmbg_model = [None for _ in range(num_models)]

models_tts = []
models_vc = []
models_info = [
    {
        "title": "Trilingual",
        "languages": ['日本語', '简体中文', 'English', 'Mix'],
        "description": """
    This model is trained on a mix up of Umamusume, Genshin Impact, Sanoba Witch & VCTK voice data to learn multilanguage.
    All characters can speak English, Chinese & Japanese.\n\n
    To mix multiple languages in a single sentence, wrap the corresponding part with language tokens
     ([JA] for Japanese, [ZH] for Chinese, [EN] for English), as shown in the examples.\n\n
    这个模型在赛马娘，原神，魔女的夜宴以及VCTK数据集上混合训练以学习多种语言。
    所有角色均可说中日英三语。\n\n
    若需要在同一个句子中混合多种语言，使用相应的语言标记包裹句子。
    （日语用[JA], 中文用[ZH], 英文用[EN]），参考Examples中的示例。
    """,
        "model_path": "./pretrained_models/G_trilingual.pth",
        "config_path": "./configs/uma_trilingual.json",
        "examples": [['你好，训练员先生，很高兴见到你。', '草上飞 Grass Wonder (Umamusume Pretty Derby)', '简体中文', 1, False],
                     ['To be honest, I have no idea what to say as examples.', '派蒙 Paimon (Genshin Impact)', 'English',
                      1, False],
                     ['授業中に出しだら，学校生活終わるですわ。', '綾地 寧々 Ayachi Nene (Sanoba Witch)', '日本語', 1, False],
                     ['[JA]こんにちわ。[JA][ZH]你好！[ZH][EN]Hello![EN]', '綾地 寧々 Ayachi Nene (Sanoba Witch)', 'Mix', 1, False]],
        "onnx_dir": "./ONNX_net/G_trilingual/"
    },
    {
        "title": "Japanese",
        "languages": ["Japanese"],
        "description": """
                       This model contains 87 characters from Umamusume: Pretty Derby, Japanese only.\n\n
                       这个模型包含赛马娘的所有87名角色，只能合成日语。
                       """,
        "model_path": "./pretrained_models/G_jp.pth",
        "config_path": "./configs/uma87.json",
        "examples": [['お疲れ様です，トレーナーさん。', '无声铃鹿 Silence Suzuka (Umamusume Pretty Derby)', 'Japanese', 1, False],
                     ['張り切っていこう！', '北部玄驹 Kitasan Black (Umamusume Pretty Derby)', 'Japanese', 1, False],
                     ['何でこんなに慣れでんのよ，私のほが先に好きだっだのに。', '草上飞 Grass Wonder (Umamusume Pretty Derby)', 'Japanese', 1, False],
                     ['授業中に出しだら，学校生活終わるですわ。', '目白麦昆 Mejiro Mcqueen (Umamusume Pretty Derby)', 'Japanese', 1, False],
                     ['お帰りなさい，お兄様！', '米浴 Rice Shower (Umamusume Pretty Derby)', 'Japanese', 1, False],
                     ['私の処女をもらっでください！', '米浴 Rice Shower (Umamusume Pretty Derby)', 'Japanese', 1, False]],
        "onnx_dir": "./ONNX_net/G_jp/"
    },
]

# synthesize voice function
def create_tts_fn(model, hps, speaker_ids):
    global tts_fn
    def tts_fn(text, speaker, language, speed, is_symbol):
        if text == None:
            raise gr.Error("Please generate a chat response in the chatbot first to generate audio.")
        if limitation:
            text_len = len(re.sub("\[([A-Z]{2})\]", "", text))
            max_len = 150
            if is_symbol:
                max_len *= 3
            if text_len > max_len:
                return "Error: Text is too long", None
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps, is_symbol)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn


def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, input_audio):
        if input_audio is None:
            return "You need to upload an audio", None
        sampling_rate, audio = input_audio
        duration = audio.shape[0] / sampling_rate
        if limitation and duration > 30:
            return "Error: Audio is too long", None
        original_speaker_id = speaker_ids[original_speaker]
        target_speaker_id = speaker_ids[target_speaker]

        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != hps.data.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
        with no_grad():
            y = torch.FloatTensor(audio)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False)
            spec_lengths = LongTensor([spec.size(-1)])
            sid_src = LongTensor([original_speaker_id])
            sid_tgt = LongTensor([target_speaker_id])
            audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
        del y, spec, spec_lengths, sid_src, sid_tgt
        return (hps.data.sampling_rate, audio)

    return vc_fn


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def create_to_symbol_fn(hps):
    def to_symbol_fn(is_symbol_input, input_text, temp_text):
        return (_clean_text(input_text, hps.data.text_cleaners), input_text) if is_symbol_input \
            else (temp_text, temp_text)

    return to_symbol_fn

parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
args = parser.parse_args()
for info in models_info:
    name = info['title']
    lang = info['languages']
    examples = info['examples']
    config_path = info['config_path']
    model_path = info['model_path']
    description = info['description']
    onnx_dir = info["onnx_dir"]
    hps = utils.get_hparams_from_file(config_path)
    model = ONNXVITS_infer.SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        ONNX_dir=onnx_dir,
        **hps.model)
    utils.load_checkpoint(model_path, model, None)
    model.eval()
    model.to(device)
    speaker_ids = hps.speakers
    speakers = list(hps.speakers.keys())
    models_tts.append((name, description, speakers, lang, examples,
                       hps.symbols, create_tts_fn(model, hps, speaker_ids),
                       create_to_symbol_fn(hps)))
    models_vc.append((name, description, speakers, create_vc_fn(model, hps, speaker_ids)))


# load chatbot pipe and send to gpu
pipe_chatbot = pipeline("text-generation", model="vicgalle/Roleplay-Llama-3-8B", device=device)

# chatbot function
def chatbot_infer(prompt, chat_history, role):
    global pipe_chatbot
    if pipe_chatbot == None:
        pipe_chatbot = pipeline("text-generation", model="vicgalle/Roleplay-Llama-3-8B", device=device)
        
    messages = [
        {"role": role, "content": prompt},
    ]
    response = pipe_chatbot(messages, max_new_tokens=250, do_sample=True)[0]["generated_text"][1]["content"]
    
    return response

def mult_thread_chatbot_infer(prompt, chat_history, role):
    
    # conduct multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        future = executor.submit(chatbot_infer, prompt, chat_history, role)
        response = future.result()
    return response

# remove the generated chat response related to action to generate audioes
def remove_between_asterisks(text):
    while True:
        start_index = text.find('*')
        if start_index == -1:
            break

        end_index = text.find('*', start_index + 1)
        if end_index == -1:
            break

        text = text[:start_index] + text[end_index + 1:]

    return text


def mult_thread_voice_infer(chatbot, speaker, language="English", speed=1.0, is_symbol=False):
    
    # conduct multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        tts_fn = models_tts[0][6]
        text = chatbot[-1][-1]
        text = remove_between_asterisks(text)
        future = executor.submit(tts_fn, text, speaker, language, speed, is_symbol)
        _, audio = future.result()
    return audio

def gen_batch_img(pipe, height, width, prompt_embeds, negative_prompt_embeds, num_inference_steps, input_img=None, strength=1.0, control_image=None, mask_image=None, num_images=4):
    res = []
    for x in range(num_images):
        i = 0
        res_image = Image.fromarray(np.zeros((64, 64)))
        while not res_image.getbbox() and i < 10:
            if isinstance(pipe, StableDiffusionReferencePipeline):
                res_image = pipe(ref_image=input_img,
                      prompt_embeds=prompt_embeds,
                      negative_prompt_embeds=negative_prompt_embeds,
                      num_inference_steps=num_inference_steps,
                      reference_attn=True,
                      reference_adain=True).images[0]
            elif isinstance(pipe, StableDiffusionControlNetInpaintPipeline):
                res_image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=input_img, strength=strength, num_inference_steps=num_inference_steps, control_image=control_image, mask_image=mask_image).images[0]
            elif input_img is not None:
                res_image = pipe(height=height, width=width, image=input_img, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_inference_steps=40).images[0]
            else:
                res_image = pipe(height=height, width=width, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_inference_steps=num_inference_steps).images[0]
            i+=1
        res.append(res_image)
    return res

def text_to_anime(prompt, negative_prompt, height, width, num_images=4):
    global pipe_txt2img
    
    # load model
    if (pipe_txt2img == None):
        gr.Info("Loading model")

        # load pipe
        pipe_txt2img = DiffusionPipeline.from_pretrained(
            "shellypeng/animever10-god-model", 
            torch_dtype=torch.float16, token=HF_TOKEN
        )
        
        # load LoRAs
        pipe_txt2img.load_lora_weights("shellypeng/detail-tweaker")
        pipe_txt2img.fuse_lora(lora_scale=0.5)
        pipe_txt2img.load_lora_weights("shellypeng/lora-eyes")
        pipe_txt2img.fuse_lora(lora_scale=0.3)

        # load textual inversions
        pipe_txt2img.load_textual_inversion("shellypeng/badhandv4")
        pipe_txt2img.load_textual_inversion("shellypeng/bad-prompt")
        pipe_txt2img.load_textual_inversion("./EasyNegative.pt")
        pipe_txt2img.load_textual_inversion("shellypeng/deepnegative")
        pipe_txt2img.load_textual_inversion("shellypeng/verybadimagenegative")
        
        # load scheduler
        pipe_txt2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_txt2img.scheduler.config, use_karras_sigmas=True)
        
        # send to gpu
        pipe_txt2img.to(device)
        gr.Info("Finished loading model")
    if height % 8:
        raise gr.Error("Please input a height with a value of multiple of 8 on the slider.")
        
    if width % 8:
        raise gr.Error("Please input a width with a value of multiple of 8 on the slider.")
        
    # prompt weighter to add weights to prompts
    compel_proc = Compel(tokenizer=pipe_txt2img.tokenizer, text_encoder=pipe_txt2img.text_encoder)
  
    # positive prompt with weights
    prompt = prompt + hidden_booster_text 

    prompt_embeds = compel_proc(prompt)

    # negative prompt with weights
    negative_prompt = negative_prompt + hidden_negative
    negative_prompt_embeds = compel_proc(negative_prompt)

    # generate result image(s)
    res = gen_batch_img(pipe_txt2img, height, width, prompt_embeds, negative_prompt_embeds, 40, num_images=num_images)

    return res




def mult_thread_txt2img(prompt_box, negative_prompt_box, height, width, num_image=4):
    
    # conduct multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        future = executor.submit(text_to_anime, prompt_box, negative_prompt_box, height, width, num_image)
        
        # case for input image for video generation
        if(num_image == 1):
            image1 = future.result()
            return image1
        
        # case for character prototype generation
        elif num_image == 4:
            image1, image2, image3, image4 = future.result()
            
            return image1, image2, image3, image4 
        
def scribble_to_image(prompt, negative_prompt, input_img, height, width):
    global pipe_scribble, hed, controlnet_scribble
    
    # load model
    if (pipe_scribble == None):
        print_msg(LOAD_MODEL_OPTION)
        
        # load HED processor for preprocessing of the ControlNet
        hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
        controlnet_scribble = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16, use_safetensors=True)
        
        # load pipe
        pipe_scribble = StableDiffusionControlNetPipeline.from_pretrained("shellypeng/animever10-god-model", controlnet=controlnet_scribble, torch_dtype=torch.float16)

        # load LoRAs
        pipe_scribble.load_lora_weights("shellypeng/detail-tweaker")
        pipe_scribble.fuse_lora(lora_scale=0.5)
        pipe_scribble.load_lora_weights("shellypeng/lora-eyes")
        pipe_scribble.fuse_lora(lora_scale=0.6)

        # load textual inversions
        pipe_scribble.load_textual_inversion("shellypeng/badhandv4")
        pipe_scribble.load_textual_inversion("shellypeng/bad-prompt")
        pipe_scribble.load_textual_inversion("shellypeng/deepnegative")
        pipe_scribble.load_textual_inversion("shellypeng/verybadimagenegative")
        pipe_scribble.load_textual_inversion("./EasyNegative.pt")

        # load scheduler
        pipe_scribble.scheduler = DPMSolverMultistepScheduler.from_config(pipe_scribble.scheduler.config, use_karras_sigmas=True)
        
        # send to gpu
        pipe_scribble.to(device)
        print_msg(FINISH_MODEL_LOADING_OPTION)
        
    if input_img is None:
        raise gr.Error("Please provide a input image.")
        
    if height % 8:
        raise gr.Error("Please input a height with a value of multiple of 8 on the slider.")
        
    if width % 8:
        raise gr.Error("Please input a width with a value of multiple of 8 on the slider.")
    
    # preprocessing input image
    input_img = np.array(input_img)
    input_img[input_img > 100] = 255
    input_img = Image.fromarray(input_img)
    input_img = hed(input_img, scribble=True)
    input_img.save("hed_img.png")
    input_img = load_image(input_img)
    
    # prompt weighter to add weights to prompts
    compel_proc = Compel(tokenizer=pipe_scribble.tokenizer, text_encoder=pipe_scribble.text_encoder)

    # positive prompt with weights
    prompt = prompt + hidden_booster_text
    prompt_embeds = compel_proc(prompt)

    # negative prompt with weights
    negative_prompt = negative_prompt + hidden_negative
    negative_prompt_embeds = compel_proc(negative_prompt)

    # generating 4 result images
    res = gen_batch_img(pipe_scribble, height, width, prompt_embeds, negative_prompt_embeds, 40, input_img, 4)
    return res

def mult_thread_scribble(prompt_box, negative_prompt_box, image_box, height, width):
    
    # conduct multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        future = executor.submit(scribble_to_image, prompt_box, negative_prompt_box, image_box, height=height, width=width)
        image1, image2, image3, image4 = future.result()
    return [image1, image2, image3, image4]

def mult_thread_live_scribble(prompt_box, negative_prompt_box, image_box, height, width):
    # get the scribbled layer of the input image
    image_box = image_box["composite"]
    
    # conduct multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        future = executor.submit(scribble_to_image, prompt_box, negative_prompt_box, image_box, height=height, width=width)
        image1, image2, image3, image4 = future.result()
    return [image1, image2, image3, image4]

def real_img2img_to_anime(prompt, negative_prompt, input_img, height, width):
    global pipe_img2img
    
    if input_img is None:
        raise gr.Error("Please provide a input image.")
    
    if height % 8:
        raise gr.Error("Please input a height with a value of multiple of 8 on the slider.")
        
    if width % 8:
        raise gr.Error("Please input a width with a value of multiple of 8 on the slider.")
    
    # preprocessing input image
    input_img = load_image(input_img)
    
    # load model
    if (pipe_img2img == None):
        print_msg(LOAD_MODEL_OPTION)
        
        # load pipe
        pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained("shellypeng/animever10-god-model",
                                                                               torch_dtype=torch.float16)
        
        # load LoRAs
        pipe_img2img.load_lora_weights("shellypeng/detail-tweaker")
        pipe_img2img.fuse_lora(lora_scale=0.5)
        pipe_img2img.load_lora_weights("shellypeng/lora-eyes")
        pipe_img2img.fuse_lora(lora_scale=0.6)

        # load textual inversions
        pipe_img2img.load_textual_inversion("shellypeng/badhandv4")
        pipe_img2img.load_textual_inversion("shellypeng/bad-prompt")
        pipe_img2img.load_textual_inversion("shellypeng/deepnegative")
        pipe_img2img.load_textual_inversion("shellypeng/verybadimagenegative")
        pipe_img2img.load_textual_inversion("./EasyNegative.pt")

        # load scheduler
        pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config, use_karras_sigmas=True)
        
        # send to gpu
        pipe_img2img.to(device)
        print_msg(FINISH_MODEL_LOADING_OPTION)


    # prompt weighter to add weights to prompts
    compel_proc = Compel(tokenizer=pipe_img2img.tokenizer, text_encoder=pipe_img2img.text_encoder)

    # positive prompt with weights
    prompt = prompt + hidden_booster_text
    prompt_embeds = compel_proc(prompt)

    # negative prompt with weights
    negative_prompt = negative_prompt + hidden_negative
    negative_prompt_embeds = compel_proc(negative_prompt)

    # generating 4 result images
    res = gen_batch_img(pipe_img2img, height, width, prompt_embeds, negative_prompt_embeds, 40, input_img, 4)

    return res

def mult_thread_img2img(prompt_box, negative_prompt_box, image_box, height, width):
    
    # conduct multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        future = executor.submit(real_img2img_to_anime, prompt_box, negative_prompt_box, image_box, height=height, width=width)
        image1, image2, image3, image4 = future.result()
    return [image1, image2, image3, image4]


def pose_to_anime(prompt, negative_prompt, image):
    global pipe_pose
    
    if image is None:
        raise gr.Error("Please provide a input image.")

    # load image
    image = load_image(image)
    
    # load model
    if (pipe_pose == None):
        print_msg(LOAD_MODEL_OPTION)
        
        # load pipe
        pipe_pose = StableDiffusionReferencePipeline.from_pretrained(
            "shellypeng/animever10-god-model", 
            torch_dtype=torch.float16, token=HF_TOKEN
        )

        # load LoRAs
        pipe_pose.load_lora_weights("shellypeng/detail-tweaker")
        pipe_pose.fuse_lora(lora_scale=0.5)
        pipe_pose.load_lora_weights("shellypeng/lora-eyes")
        pipe_pose.fuse_lora(lora_scale=0.6)

        # load textual inversions
        pipe_pose.load_textual_inversion("shellypeng/badhandv4")
        pipe_pose.load_textual_inversion("shellypeng/bad-prompt")
        pipe_pose.load_textual_inversion("shellypeng/deepnegative")
        pipe_pose.load_textual_inversion("shellypeng/verybadimagenegative")
        pipe_pose.load_textual_inversion("./EasyNegative.pt")
        pipe_pose.scheduler = DPMSolverMultistepScheduler.from_config(pipe_pose.scheduler.config, use_karras_sigmas=True)

        # send to gpu
        pipe_pose.to(device)
        print_msg(FINISH_MODEL_LOADING_OPTION)

    # prompt weighter to add weights to prompts
    compel_proc = Compel(tokenizer=pipe_pose.tokenizer, text_encoder=pipe_pose.text_encoder)

    # positive prompt with weights
    prompt = prompt + hidden_booster_text
    prompt_embeds = compel_proc(prompt)

    # negative prompt with weights
    negative_prompt = negative_prompt + hidden_vary_negative +hidden_negative
    negative_prompt_embeds = compel_proc(negative_prompt)

    # generating 4 result images
    res = gen_batch_img(pipe_pose, height, width, prompt_embeds, negative_prompt_embeds, 40, image, 4)

    return res

def mult_thread_pose2img(prompt_box, negative_prompt_box, image_box):
    
    # conduct multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        future = executor.submit(pose_to_anime, prompt_box, negative_prompt_box, image_box)
        image1, image2, image3, image4 = future.result()
    return [image1, image2, image3, image4]

# preprocess for inpainting
def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


# fills the inpainted outline
def fill_mask(mask_image):
    # convert mask to numpy array
    mask_image = np.array(mask_image)
    
    # obtain contours of enclosed shapes
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # fill the enclosed contours with white color
    cv2.drawContours(mask_image, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    mask_image = Image.fromarray(mask_image)
    
    return mask_image

def inpaint(prompt, negative_prompt, image, btn):
    global pipe_inpaint
    
    # load model
    if (pipe_inpaint == None):
        print_msg(LOAD_MODEL_OPTION)
        
        # load ControlNet for inpainting
        controlnet_inpaint = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16")
        
        # load pipe
        pipe_inpaint = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "shellypeng/animever10-god-model", controlnet=controlnet_inpaint, torch_dtype=torch.float16
        )
        
        # load LoRAs
        pipe_inpaint.load_lora_weights("shellypeng/detail-tweaker")
        pipe_inpaint.fuse_lora(lora_scale=0.5)
        pipe_inpaint.load_lora_weights("shellypeng/lora-eyes")
        pipe_inpaint.fuse_lora(lora_scale=0.6)

        # load textual inversions
        pipe_inpaint.load_textual_inversion("shellypeng/badhandv4")
        pipe_inpaint.load_textual_inversion("shellypeng/bad-prompt")
        pipe_inpaint.load_textual_inversion("shellypeng/deepnegative")
        pipe_inpaint.load_textual_inversion("shellypeng/verybadimagenegative")
        pipe_inpaint.load_textual_inversion("./EasyNegative.pt")
        pipe_inpaint.scheduler = DPMSolverMultistepScheduler.from_config(pipe_inpaint.scheduler.config, use_karras_sigmas=True)
        
        # send to gpu
        pipe_inpaint.to(device)
        print_msg(FINISH_MODEL_LOADING_OPTION)
        
    
    compel_proc = Compel(tokenizer=pipe_inpaint.tokenizer, text_encoder=pipe_inpaint.text_encoder)

    # positive prompt with weights
    prompt = prompt + hidden_booster_text
    prompt_embeds = compel_proc(prompt)

    # negative prompt with weights
    negative_prompt = negative_prompt + hidden_negative
    negative_prompt_embeds = compel_proc(negative_prompt)
    
    # set strength - how dissimilar to the reference image of the generated images
    strength = 0.9
    
    # load the mask layer of input image
    mask_image = load_image(image["layers"][0])
    
    if image is None or mask_image is None:
        raise gr.Error("Please provide an input image.")
    
    # convert to numpy for preprocessing
    mask_image = np.array(mask_image)
    
    # convert user's mask outline to white for ControlNet Inpaint Pipeline to recognize(it can only recognize monochrome images)
    mask_image[np.all(mask_image == [93, 63, 211], axis=-1)] = [255, 255, 255]
    mask_image = Image.fromarray(mask_image).convert('L')

    # fill the enclosed outline the user has drawn
    mask_image = fill_mask(mask_image)
    image = load_image(image["background"])
    image_shape = image.size
    mask_image.resize(image_shape)

    # make the control image
    control_image = make_inpaint_condition(image, mask_image)

    # generating 4 result images
    res = gen_batch_img(pipe_inpaint, height, width, prompt_embeds, negative_prompt_embeds, 40, image, strength, control_image, mask_image, 4)

    return res

def mult_thread_inpaint(prompt_box, negative_prompt_box, image, btn):
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        future = executor.submit(inpaint, prompt_box, negative_prompt_box, image, btn)
        image1, image2, image3, image4 = future.result()
    return [image1, image2, image3, image4]

def chibi(prompt, negative_prompt, image, height, width):
    global pipe_chibi
    
    if image is None:
        raise gr.Error("Please provide an input image.")
        
    if height % 8:
        raise gr.Error("Please input a height with a value of multiple of 8 on the slider.")
        
    if width % 8:
        raise gr.Error("Please input a width with a value of multiple of 8 on the slider.")
    
    # preprocessing input image
    image = load_image(image)
    
    # load model
    if pipe_chibi == None:
        print_msg(LOAD_MODEL_OPTION)
        
        # load pipe
        pipe_chibi = StableDiffusionImg2ImgPipeline.from_pretrained("shellypeng/animever10-god-model",
                                                                       torch_dtype=torch.float16, token=HF_TOKEN)

        # load LoRAs
        pipe_chibi.load_lora_weights("shellypeng/detail-tweaker")
        pipe_chibi.fuse_lora(lora_scale=0.5)
        pipe_chibi.load_lora_weights("shellypeng/chibi-artstyle")
        pipe_chibi.fuse_lora(lora_scale=2.0)

        # load textual inversions
        pipe_chibi.load_textual_inversion("shellypeng/badhandv4")
        pipe_chibi.load_textual_inversion("shellypeng/bad-prompt")
        pipe_chibi.load_textual_inversion("shellypeng/deepnegative")
        pipe_chibi.load_textual_inversion("shellypeng/verybadimagenegative")
        pipe_chibi.load_textual_inversion("./EasyNegative.pt")
        pipe_chibi.scheduler = DPMSolverMultistepScheduler.from_config(pipe_chibi.scheduler.config, use_karras_sigmas=True)
        
        # send to gpu
        pipe_chibi.to(device)
        print_msg(FINISH_MODEL_LOADING_OPTION)
        

    # prompt weighter to add weights to prompts
    compel_proc = Compel(tokenizer=pipe_chibi.tokenizer, text_encoder=pipe_chibi.text_encoder)

    # positive prompt with weights
    prompt = "chibi+++" + prompt + hidden_booster_text
    prompt_embeds = compel_proc(prompt)

    # negative prompt with weights
    negative_prompt = negative_prompt + hidden_negative
    negative_prompt_embeds = compel_proc(negative_prompt)

    # generating 4 result images
    res = gen_batch_img(pipe_chibi, height, width, prompt_embeds, negative_prompt_embeds, 40, image, strength=0.5, num_images=4)

    return res

def mult_thread_chibi(prompt_box, negative_prompt_box, image_box, height, width):
   
    # conduct multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        future = executor.submit(chibi, prompt_box, negative_prompt_box, image_box, height=height, width=width)
        image1, image2, image3, image4 = future.result()
    return [image1, image2, image3, image4]

def lineart(prompt, negative_prompt, lineart_image):
    global pipe_lineart, lineart_processor
    
    if lineart_image is None:
        raise gr.Error("Please provide a lineart image.")
    
    # load model
    if pipe_lineart == None:
        print_msg(LOAD_MODEL_OPTION)
        
        # load lineart processor for preprocessing of the ControlNet
        lineart_processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        controlnet_lineart = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15s2_lineart_anime", torch_dtype=torch.float16, token=HF_TOKEN)
        
        # load pipe
        pipe_lineart = StableDiffusionControlNetPipeline.from_pretrained(
            "shellypeng/animever10-god-model", controlnet=controlnet_lineart,
            torch_dtype=torch.float16, token=HF_TOKEN
        )
        
        # load LoRAs
        pipe_lineart.load_lora_weights("shellypeng/detail-tweaker")
        pipe_lineart.fuse_lora(lora_scale=0.5)
        pipe_lineart.load_lora_weights("shellypeng/lora-eyes")
        pipe_lineart.fuse_lora(lora_scale=0.3)
        
        # load textual inversions
        pipe_lineart.load_textual_inversion("shellypeng/badhandv4")
        pipe_lineart.load_textual_inversion("shellypeng/bad-prompt")
        pipe_lineart.load_textual_inversion("shellypeng/deepnegative")
        pipe_lineart.load_textual_inversion("shellypeng/verybadimagenegative")
        pipe_lineart.load_textual_inversion("./EasyNegative.pt")
        pipe_lineart.scheduler = DPMSolverMultistepScheduler.from_config(pipe_lineart.scheduler.config, use_karras_sigmas=True)
        
        # send to gpu
        pipe_lineart.to(device)
        print_msg(FINISH_MODEL_LOADING_OPTION)
    
    # preprocessing input image
    lineart_image = load_image(lineart_image)
    width, height = lineart_image.size
    lineart_image = lineart_processor(lineart_image)

    # prompt weighter to add weights to prompts
    compel_proc = Compel(tokenizer=pipe_lineart.tokenizer, text_encoder=pipe_lineart.text_encoder)

    # positive prompt with weights
    prompt = prompt + hidden_booster_text 
    prompt_embeds = compel_proc(prompt)

    # negative prompt with weights
    negative_prompt = negative_prompt + hidden_negative
    negative_prompt_embeds = compel_proc(negative_prompt)

    # generating 4 result images
    res = gen_batch_img(pipe_lineart, height, width, prompt_embeds, negative_prompt_embeds, 40, lineart_image, num_images=4)
    return res

def mult_thread_lineart(prompt_box, negative_prompt_box, image_box):
    
    # conduct multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        future = executor.submit(lineart, prompt_box, negative_prompt_box, image_box)
        image1, image2, image3, image4 = future.result()
    return [image1, image2, image3, image4]

# get a mask for background removal
def get_mask(img, s=1024):
    img = (img / 255).astype(np.float32)
    h, w = h0, w0 = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = rmbg_model.run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask

def rmbg_fn(img):
    global rmbg_model
    
    if img is None:
        raise gr.Error("Please provide an input image.")
    
    # load model
    if rmbg_model == None:
        print_msg(LOAD_MODEL_OPTION)
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model_path = huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
        rmbg_model = rt.InferenceSession(model_path, providers=providers)
        print_msg(FINISH_MODEL_LOADING_OPTION)
    
    # remove background
    img = np.array(img)
    mask = get_mask(img)
    img = (mask * img + 255 * (1 - mask)).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    img = np.concatenate([img, mask], axis=2, dtype=np.uint8)
    mask = mask.repeat(3, axis=2)
    img = Image.fromarray(img)
    return img

def mult_thread_remove_bg(image_box):
    
    # conduct multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        future = executor.submit(rmbg_fn, image_box)
        image = future.result()
    return image

def I2VGen_video(prompt, negative_prompt, height, width, num_inference_steps, num_frames, video_image, fps=30):
    global pipe_video
    
    if video_image == None:
        raise gr.Error("Please provide a input image.")
        
    if height % 8:
        raise gr.Error("Please input a height with a value of multiple of 8 on the slider.")
        
    if width % 8:
        raise gr.Error("Please input a width with a value of multiple of 8 on the slider.")
    
    # get attributes - height, size, width - from input image
    size = video_image.size
    height = video_image.size[1]
    width = video_image.size[0]
    
    # load model
    if (pipe_video == None):
        print_msg(LOAD_MODEL_OPTION)
        
        # load pipe
        pipe_video = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16)
        pipe_video.to(device) 
        print_msg(FINISH_MODEL_LOADING_OPTION)
        
    # generate result video
    output = pipe_video(
        prompt=prompt+hidden_booster_text,
        negative_prompt=negative_prompt+hidden_negative,
        image=video_image,
        target_fps=fps,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        height=size[1],
        width=size[0]
    ).frames[0]
    
    return output

def mult_thread_I2VGen_video(prompt_box, negative_prompt_box, height, width, num_inference_steps, num_frames, video_image, fps):
    
    # conduct multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=12000) as executor:
        future = executor.submit(I2VGen_video, prompt_box, negative_prompt_box, height, width, num_inference_steps, num_frames, video_image, fps)
        video1 = future.result()
    
    # save video for preview on Gradio app interface
    export_to_gif(video1, "I2VGen_video1.gif")
    clip = mp.VideoFileClip("I2VGen_video1.gif")
    out_file_name = "out.mp4"
    clip.write_videofile(out_file_name)
    return out_file_name

# delete models to release memory
def delete_models(button):
    global pipe_chatbot, pipe_scribble, pipe_txt2img, pipe_img2img, pipe_lineart, pipe_pose, pipe_inpaint, pipe_video, pipe_chibi, rmbg_model
    
    # check which model the user wishes to delete
    if pipe_chatbot is not None and button == "Chatbot - 50%":
        del pipe_chatbot
    if pipe_scribble is not None and button == "Scribble to Image Generator - 7%":
        del pipe_scribble
    if pipe_video is not None and button == "Video Generator - 7%":
        del pipe_video
    if pipe_txt2img is not None and button == "Text to Image Generator - 5%":
        del pipe_txt2img
    if pipe_img2img is not None and button == "Image to Image Generator - 5%":
        del pipe_img2img
    if pipe_lineart is not None and button == "Lineart to Image Generator - 5%":
        del pipe_lineart
    if pipe_pose is not None and button == "Pose Variation Generator - 5%":
        del pipe_pose
    if pipe_inpaint is not None and button == "Inpainting Generator - 5%":
        del pipe_inpaint
    if pipe_chibi is not None and button == "Chibi Generator - 5%":
        del pipe_chibi
    if rmbg_model is not None and button == "Remove Background Generator - 5%":
        del rmbg_model
    
    # print message when finished deleting models
    gr.Info("Model deleted.")

# transport selected image in gallery to other tabs in app
def get_select_image(prompt, negative_prompt, evt: gr.SelectData):
    return evt.value["image"]["path"], prompt, negative_prompt

# Build the app UI interface
theme = gr.themes.Soft()
with gr.Blocks(theme=theme, css="""footer {visibility: hidden}""", title="AniPack") as iface:
    # audio player for playing congratulation voice messages
    audio = gr.Audio(autoplay=True, visible=False)
    
    # tab for creating prototype
    with gr.Tab("Create Prototype"):
        gr.Markdown(
            """
            # <b>AniPack
            Welcome to AniPack – your personalized anime companion creator - with dialogue, images, and videos.</b>
            
            AniPack is a collection of tools for creating a personalized anime character as your friendly companion, which enables chatting, image generation and video instantiation. Switch to the corresponding tabs to explore.
            
            Tip: Click on the generated images to send the selected image convenienty to next step; use ++ and -- to emphasize or weaken the a prompt word to make it more/less influential to the generation.
            
            Note: First generations of each tab may be slow. Subsequent generations will be faster.
            """
        )
        gr.Markdown(
            """
            # Build your companion's prototype
            """
        )
        
        # get inputs for creating character prototype
        with gr.Row():
            prompt_box = gr.Textbox(label="Prompt", placeholder="Enter a prompt", lines=3)
            height = gr.Slider(512, 1024, label="Height", step=8, value=800)
        with gr.Row():
            neg_prompt_box = gr.Textbox(label="Negative Prompt", placeholder="Enter a negative prompt(things you don't want to include in the generated image)", lines=3)
            width = gr.Slider(512, 1024, label="Width", step=8, value=640)
        txt2img_gen_btn = gr.Button(value="Generate With Text")

        # sub-tab for image-to-image generation
        with gr.Tab("Upload Image"):
            # get inputs
            img2img_image_box = gr.Image(label="Input Image", height=500, type='pil')
            img2img_gen_btn = gr.Button(value="Generate With Image")
        
        # sub-tab for scribble-to-image generation
        with gr.Tab("Upload Scribble Image"):
            gr.Markdown(
                """
                Please upload an image drawn on digital board(e.g. laptop, drawing pad) with black scribbles and white background, with only black and white colors.
                PS: Scribbles with brush size = 2px comes out with best effect.
                """
            )
            scribble_bg = Image.open("sketch_bg.png")
            # get inputs
            scribble2img_image_box = gr.Image(value=scribble_bg, label="Input Image", height=500, type='pil')
            scribble2img_gen_btn = gr.Button(value="Generate With Scribbles")
            
        # sub-tab for live scribble-to-image generation
        with gr.Tab("Draw Scribbles"):
            # get inputs
            live_scribble2img_image_box = gr.Sketchpad(value=scribble_bg, label="Draw Scribbles", type='pil', brush=gr.Brush(default_size="2", color_mode="fixed", colors=["#000000"]))
            live_scribble2img_gen_btn = gr.Button(value="Generate With Scribbles")
        
        # sub-tab for lineart-to-image generation
        with gr.Tab("Colorize lineart"):
            gr.Markdown(
                """
                Color your anime lineart with text prompts.
                """
            )
            # get inputs
            lineart_image_box = gr.Image(label="Lineart Image", height=512, type='pil')
            lineart_gen_btn = gr.Button(value="Colorize Lineart")

        # gallery to show generated images
        gallery = gr.Gallery(
            label="Generated images", show_label=True, elem_id="gallery"
        , columns=[4], rows=[1], object_fit="contain", height=512, allow_preview=True)

        # handle on-click events - generate images
        txt2img_gen_btn.click(mult_thread_audio, [], [audio])
        txt2img_gen_btn.click(fn=mult_thread_txt2img, inputs=[prompt_box, neg_prompt_box, height, width], outputs=[gallery])
        
        img2img_gen_btn.click(mult_thread_audio, [], [audio])
        img2img_gen_btn.click(fn=mult_thread_img2img, inputs=[prompt_box, neg_prompt_box, img2img_image_box, height, width], outputs=[gallery])
        
        scribble2img_gen_btn.click(mult_thread_audio, [], [audio])
        scribble2img_gen_btn.click(fn=mult_thread_scribble, inputs=[prompt_box, neg_prompt_box, scribble2img_image_box, height, width], outputs=[gallery])
        
        live_scribble2img_gen_btn.click(mult_thread_audio, [], [audio])
        live_scribble2img_gen_btn.click(fn=mult_thread_live_scribble, inputs=[prompt_box, neg_prompt_box, live_scribble2img_image_box, height, width], outputs=[gallery])
        
        lineart_gen_btn.click(mult_thread_audio, [], [audio])
        lineart_gen_btn.click(fn=mult_thread_lineart, inputs=[prompt_box, neg_prompt_box, lineart_image_box], outputs=[gallery])
    
    # tab for chatbot
    with gr.Tab("Chat"):    
        gr.Markdown(
            """
            # Chat with a personalized anime companion with a personality and look of your choice.
            
            In "Personality", enter the companion's characteristics, e.g. a tsundere girl, a lively boy, an elegant lady. Then upload a preferred image of your companion and can then start chatting!
            """
        )
        
        # get inputs for chatbot
        with gr.Row(equal_height=True):
            chatbot_input_img = gr.Image(label="Companion Image", interactive=True, type='pil')
            
            with gr.Column():
                placeholder = gr.Textbox(visible=False)
                chatbot_personality_box = gr.Textbox(label="Personality", placeholder="Enter the personality of your companion", lines=1)
                char_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label="Select Companion's Voice")
                chatbot_ = gr.Chatbot(height=350, render=False)
                chatbot = gr.ChatInterface(
                    chatbot_infer,
                    chatbot=chatbot_,
                    additional_inputs=[chatbot_personality_box]
                )
                with gr.Row(equal_height=True):
                    audio_output = gr.Audio(label="Output Audio", autoplay=True)
                    audio_gen_btn = gr.Button("Generate Audio")
                    audio_gen_btn.click(mult_thread_voice_infer,
                              inputs=[chatbot_, char_dropdown],
                              outputs=[audio_output])
                
        # transport selected image from prototye creation to chatbot tab
        gallery.select(get_select_image, [prompt_box, neg_prompt_box], [chatbot_input_img, placeholder, placeholder])
        
    # tab for pose variation / consistent looking character with variation
    with gr.Tab("Vary Poses"):    
        gr.Markdown(
            """
            # Vary your character's poses
            
            Best effect comes with a close match of your prompt that generated the uploaded image.
            """
        )
        
        # get inputs for variation creation
        with gr.Column():
            pose_prompt_box = gr.Textbox(label="Prompt", placeholder="Enter a prompt", lines=3, scale=1)
            pose_neg_prompt_box = gr.Textbox(label="Negative Prompt", placeholder="Enter a negative prompt(things you don't want to include in the generated image)", lines=3, scale=1)
        pose_input_img = gr.Image(label="Current Image", height=350, type='pil')
        pose_gen_btn = gr.Button(value="Vary Pose")
        
        # gallery to show generated images
        pose_gallery = gr.Gallery(
            label="Generated images", show_label=True, elem_id="gallery"
        , columns=[4], rows=[1], object_fit="contain", height=512, allow_preview=True)
        
        # handle on-click events - generate pose 
        pose_gen_btn.click(mult_thread_audio, [], [audio])
        pose_gen_btn.click(fn=mult_thread_pose2img, inputs=[pose_prompt_box, pose_neg_prompt_box, pose_input_img], outputs=[pose_gallery])
        
        # transport selected image from prototye creation to pose variation tab
        gallery.select(get_select_image, [prompt_box, neg_prompt_box], [pose_input_img, pose_prompt_box, pose_neg_prompt_box]) 
    # tab for inpainting
    with gr.Tab("Inpainting"):
        gr.Markdown(
            """
            # (Optional) Refine your companion with inpainting
            Paint the outlines of the area that you wish to modify. This keeps other parts of the image perfectly consistent.
            """
        )
        
        # get inputs for inpainting
        inpaint_prompt_box = gr.Textbox(label="Prompt", placeholder="Enter what you wish to replace the inpaint", lines=3)
        inpaint_neg_prompt_box = gr.Textbox(label="Negative Prompt", placeholder="Enter a negative prompt(things you don't want to include in the generated image)", lines=3)
        inpaint_image_box = gr.ImageEditor(interactive=True, type="pil", height=500, brush=gr.Brush(default_size="10"))
        
        # transport selected image from prototye creation to inpainting tab
        gallery.select(get_select_image,  [prompt_box, neg_prompt_box], [inpaint_image_box, inpaint_prompt_box, inpaint_neg_prompt_box])
        inpaint_btn = gr.Button(value="Inpaint")
        
        # gallery to show generated images
        inpaint_gallery = gr.Gallery(
            label="Inpainted Images", show_label=True, elem_id="gallery"
        , columns=[4], rows=[1], object_fit="contain", height=512)
        
        # handle on-click events - generate images
        inpaint_btn.click(mult_thread_audio, [], [audio])
        inpaint_btn.click(fn=mult_thread_inpaint, inputs=[inpaint_prompt_box, inpaint_neg_prompt_box, inpaint_image_box], outputs=[inpaint_gallery])
    
    # tab for video generation
    with gr.Tab("Generate Video"):
        gr.Markdown(
            """
            # Generate Video
            Create a video of your companion.
            Larger fps = smaller motion and smoother video.
            Lower fps = larger motion and more intermittent video.
            """
        )
        
        # get inputs for video generation
        with gr.Row():
            vid_prompt_box = gr.Textbox(label="Prompt", placeholder="Enter a prompt", lines=3)
            height = gr.Slider(512, 1024, label="Height", step=8, value=880)
        with gr.Row():
            vid_neg_prompt_box = gr.Textbox(label="Negative Prompt", placeholder="Enter a negative prompt(things you don't want to include in the generated image)", lines=3)
            width = gr.Slider(512, 1024, label="Width", step=8, value=640)
#             video_image_box = gr.Image(label="Reference Image", type='pil', height=512)
        video_steps = gr.Slider(1, 500, label="Inference Steps", value=100)
        with gr.Row():
            num_frames = gr.Slider(10, 40, label="Number of Frames", value=10)
            fps_slider = gr.Slider(20, 50, label="FPS", value=40)

        video_gen_btn = gr.Button(value="Generate Video")
        
        with gr.Row():
            video_image = gr.Image(label="Video Image", type='pil')
            
            video1_box = gr.Video(label="Video", height=512)
            
        # handle on-click events - generate videos
        video_gen_btn.click(mult_thread_audio, [], [audio])
        video_gen_btn.click(fn=mult_thread_I2VGen_video, inputs=[vid_prompt_box, vid_neg_prompt_box, height, width, video_steps, num_frames, video_image, fps_slider], 
                            outputs=[video1_box])
        
        # transport selected image from pose variation to video generation tab
        pose_gallery.select(get_select_image, [pose_prompt_box, pose_neg_prompt_box], [video_image, vid_prompt_box, vid_neg_prompt_box]) # try replace None by gallery

    # tab for chibi generation
    with gr.Tab("Generate Chibi"):
        gr.Markdown(
            """
            # Make a cute Chibi for your companion.
            """
        )
        
        # get inputs for chibi generation
        avatar_prompt_box = gr.Textbox(label="Prompt", placeholder="Enter a prompt", lines=3)
        avatar_height = gr.Slider(512, 1024, label="Height", step=8, visible=False)
        avatar_neg_prompt_box = gr.Textbox(label="Negative Prompt", placeholder="Enter a negative prompt(things you don't want to include in the generated image)", lines=3)
        avatar_width = gr.Slider(512, 1024, label="Width", step=8, visible=False)
        
        with gr.Row():
            avatar_ref_image = gr.Image(label="Reference Image", height=512, type='pil')
            
            # gallery to show generated images
            avatar_gallery = gr.Gallery(
            label="Generated images", show_label=True, elem_id="gallery"
            , columns=[4], rows=[1], object_fit="contain", height=512, allow_preview=True)
        
        avatar_gen_btn = gr.Button(value="Generate Chibi")
        
        # handle on-click events - generate images
        avatar_gen_btn.click(mult_thread_audio, [], [audio])

        avatar_gen_btn.click(fn=mult_thread_chibi, inputs=[avatar_prompt_box, avatar_neg_prompt_box, avatar_ref_image, avatar_height, avatar_width], outputs=[avatar_gallery])
        
        pose_gallery.select(get_select_image, [pose_prompt_box, pose_neg_prompt_box], [avatar_ref_image, avatar_prompt_box, avatar_neg_prompt_box]) # try replace None by gallery

    # tab for background removal
    with gr.Tab("Remove Background"):
        gr.Markdown(
            """
            # Remove background for your companion.
            """
        )
        
        # get inputs for background removal
        with gr.Row():
            anime_char_image = gr.Image(label="Input Image", height=512, type='pil')
            anime_char_remove_bg_image = gr.Image(label="Generated Image", height=512, type='pil')
        
        remove_bg_btn = gr.Button(value="Remove Background")
        
        # handle on-click events - generate images
        remove_bg_btn.click(mult_thread_audio, [], [audio])
        remove_bg_btn.click(fn=mult_thread_remove_bg, inputs=[anime_char_image], outputs=[anime_char_remove_bg_image])
    
    # tab for memory release
    with gr.Tab("Release Memory"):
        gr.Markdown(
            """
            # Please delete some models when encountering the "Out of Memory" error. 
            
            The percentage after each button's label determines how much memory will be gained upon the deletion. Beware that models will have to be re-loaded after deletion, thus the waiting time for the corresponding feature will be longer in the first generation of the corresponding model deleted.
            """
        )
        
        # buttons to release models
        with gr.Column():
            chatbot_release_btn = gr.Button(value="Chatbot - 50%")
            scribble2img_release_btn = gr.Button(value="Scribble to Image Generator - 7%")
            video_release_btn = gr.Button(value="Video Generator - 7%")
            txt2img_release_btn = gr.Button(value="Text to Image Generator - 5%")
            img2img_release_btn = gr.Button(value="Image to Image Generator - 5%")
            lineart2img_release_btn = gr.Button(value="Lineart to Image Generator - 5%")
            pose2img_release_btn = gr.Button(value="Pose Variation Generator - 5%")
            inpaint_release_btn = gr.Button(value="Inpainting Generator - 5%")
            chibi_release_btn = gr.Button(value="Chibi Generator - 5%")
            rmbg_release_btn = gr.Button(value="Remove Background Generator - 5%")
            
            # handle on-click events - model deletion
            chatbot_release_btn.click(fn=delete_models, inputs=[chatbot_release_btn], outputs=[])
            scribble2img_release_btn.click(fn=delete_models, inputs=[scribble2img_release_btn], outputs=[])
            video_release_btn.click(fn=delete_models, inputs=[video_release_btn], outputs=[])
            txt2img_release_btn.click(fn=delete_models, inputs=[txt2img_release_btn], outputs=[])
            img2img_release_btn.click(fn=delete_models, inputs=[img2img_release_btn], outputs=[])
            lineart2img_release_btn.click(fn=delete_models, inputs=[lineart2img_release_btn], outputs=[])
            pose2img_release_btn.click(fn=delete_models, inputs=[pose2img_release_btn], outputs=[])
            inpaint_release_btn.click(fn=delete_models, inputs=[inpaint_release_btn], outputs=[])
            chibi_release_btn.click(fn=delete_models, inputs=[chibi_release_btn], outputs=[])
            rmbg_release_btn.click(fn=delete_models, inputs=[rmbg_release_btn], outputs=[])

# function that runs the Gradio interface
def run():
    iface.launch(share=True, debug=True, inline=True)

# close any opened interface
iface.close()

# run the app interface
run()

