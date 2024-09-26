#!/usr/bin/env python
# coding: utf-8

# file: character_with_I2V.py
# -*- coding: utf-8 -*-

# importing the modules
import os
import re
import json
from random import randrange

import logging
logging.getLogger('numba').setLevel(logging.WARNING)
import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
import torch
from torch import no_grad, LongTensor
import torchvision
from torchvision.io import write_video
import torchvision.transforms as transforms
from moviepy.editor import *
import moviepy.editor as mp
import gradio as gr
from datetime import datetime
from compel import Compel
import onnxruntime as rt
import imageio
import tempfile
import librosa
import psutil

from voice import commons
from voice_synthesize_utils import *
from voice import commons
from model_base import ChatbotModel
from model_base import Txt2imgModel
from model_base import Scribble2imgModel
from model_base import Img2imgModel
from model_base import Img2poseModel
from model_base import InpaintModel
from model_base import ChibiModel
from model_base import Lineart2imgModel
from model_base import Img2videoModel
from model_base import RMBGModel


# set gpu device
device = "cuda" if torch.cuda.is_available() else "cpu"

# initialize pipeline(i.e. models) to None
num_models = 10
pipe_chatbot, pipe_txt2img, pipe_scribble, pipe_img2img, pipe_pose, pipe_inpaint, pipe_chibi, pipe_lineart, pipe_video, rmbg_model = ChatbotModel(), Txt2imgModel(), Scribble2imgModel(), Img2imgModel(), Img2poseModel(), InpaintModel(), ChibiModel(), Lineart2imgModel(), Img2videoModel(), RMBGModel()

# list of congratulation voice messages
audio_list = ["congrats-audios/audio1.wav", "congrats-audios/audio2.wav", "congrats-audios/audio3.wav", 
              "congrats-audios/audio4.wav", "congrats-audios/audio5.wav", "congrats-audios/audio6.wav", 
              "congrats-audios/audio7.wav", "congrats-audios/audio8.wav", "congrats-audios/audio9.wav", 
              "congrats-audios/audio10.wav", "congrats-audios/audio11.wav", "congrats-audios/audio12.wav", 
             ]


# print some messages on screen when certain features are triggered
def print_msg(option):
    if option == "load model":
        gr.Info("Loading model")
    elif option == "finish model loading":
        gr.Info("Finished loading model")


# plays congratulation voice messages
def play_audio():
    audio = audio_list[randrange(len(audio_list))]
    return audio


# chatbot function
def chatbot_infer(prompt, chat_history, role):
    global pipe_chatbot
    
    # load chatbot model
    if pipe_chatbot.pipe is None:
        pipe_chatbot.multi_thread_load_model()   
    
    # get chatbot response
    messages = [
        {"role": role, "content": prompt},
    ]
    response = pipe_chatbot.infer(messages=messages, max_new_tokens=250, do_sample=True)
    
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


# generate character's voice audio
def voice_infer(chatbot, speaker, language="English", speed=1.0, is_symbol=False):
    tts_fn = models_tts[0][6]
    text = chatbot[-1][-1]
    text = remove_between_asterisks(text)
    future = tts_fn(text, speaker, language, speed, is_symbol)
    _, audio = future
    return audio


def text_to_anime(prompt, negative_prompt, height, width, num_images=4):
    global pipe_txt2img
    
    # load model
    if (pipe_txt2img.pipe is None):
        pipe_txt2img.multi_thread_load_model()     
    
    # infer 
    res = pipe_txt2img.infer(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width, 
                             num_images=num_images)
    
    return res


def scribble_to_image(prompt, negative_prompt, input_image, height, width):
    global pipe_scribble, hed, controlnet_scribble
    
    # load model
    if (pipe_scribble.pipe is None):
        pipe_scribble.multi_thread_load_model()  
    
    # infer
    res = pipe_scribble.infer(prompt=prompt, negative_prompt=negative_prompt, input_img=input_image, 
                              height=height, width=width)
      
    return res


def live_scribble(prompt, negative_prompt, image_box):
    global pipe_scribble, hed, controlnet_scribble

    # get the scribbled layer of the input image
    input_image = image_box["composite"]
    
    # load model
    if (pipe_scribble.pipe is None):
        pipe_scribble.multi_thread_load_model()  
    
    # infer
    res = pipe_scribble.infer(prompt=prompt, negative_prompt=negative_prompt, input_img=input_image)

    return res
    

def real_img2img_to_anime(prompt, negative_prompt, input_image):
    global pipe_img2img
    
    # load model
    if (pipe_img2img.pipe is None):
        pipe_img2img.multi_thread_load_model()  
    
    # infer
    res = pipe_img2img.infer(prompt=prompt, negative_prompt=negative_prompt, input_img=input_image)

    return res


def pose_to_anime(prompt, negative_prompt, input_image):
    global pipe_pose
    
    # load model
    if (pipe_pose.pipe is None):
        pipe_pose.multi_thread_load_model()  
    
    # infer
    res = pipe_pose.infer(prompt=prompt, negative_prompt=negative_prompt, input_img=input_image, 
                          height=height, width=width)

    return res


def inpaint(prompt, negative_prompt, input_image, btn):
    global pipe_inpaint
    
    # load model
    if (pipe_inpaint.pipe is None):
        pipe_inpaint.multi_thread_load_model()  
    
    # infer
    res = pipe_inpaint.infer(prompt=prompt, negative_prompt=negative_prompt, input_img=input_image)

    return res


def chibi(prompt, negative_prompt, input_image, height, width):
    global pipe_chibi
    
    # load model
    if (pipe_chibi.pipe is None):
        pipe_chibi.multi_thread_load_model()  
    
    # infer
    res = pipe_chibi.infer(prompt=prompt, negative_prompt=negative_prompt, input_img=input_image)

    return res
    

def lineart(prompt, negative_prompt, input_image):
    global pipe_lineart, lineart_processor
    
    # load model
    if (pipe_lineart.pipe is None):
        pipe_lineart.multi_thread_load_model()  
    
    # infer
    res = pipe_lineart.infer(prompt=prompt, negative_prompt=negative_prompt, input_img=input_image)

    return res
    

def rmbg_fn(input_image):
    global rmbg_model
    
    # load model
    if (rmbg_model.pipe is None):
        rmbg_model.multi_thread_load_model()  
    
    # infer
    res = rmbg_model.infer(input_img=input_image)

    return res
    

def I2VGen_video(prompt, negative_prompt, num_inference_steps, num_frames, input_img, fps=30):
    global pipe_video
    
    # load model
    if (pipe_video.pipe is None):
        pipe_video.multi_thread_load_model()  
    
    # infer
    res = pipe_video.infer(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps,
                           num_frames=num_frames, input_img=input_img, 
                           fps=fps)

    return res


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
            height = gr.Slider(512, 1960, label="Height", step=8, value=1280)
        with gr.Row():
            neg_prompt_box = gr.Textbox(label="Negative Prompt", placeholder="Enter a negative prompt(things you don't want to include in the generated image)", lines=3)
            width = gr.Slider(512, 1960, label="Width", step=8, value=1000)
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
                PS: Scribbles with brush size = 4px comes out with best effect.
                """
            )
            scribble_bg = Image.open("sketch_bg.png")
            # get inputs
            scribble2img_image_box = gr.Image(value=scribble_bg, label="Input Image", height=500, type='pil')
            scribble2img_gen_btn = gr.Button(value="Generate With Scribbles")
            
        # sub-tab for live scribble-to-image generation
        with gr.Tab("Draw Scribbles"):
            # get inputs
            live_scribble2img_image_box = gr.Sketchpad(value=scribble_bg, label="Draw Scribbles", height=500, type='pil', brush=gr.Brush(default_size="2", color_mode="fixed", colors=["#000000"]))
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
        , columns=[4], rows=[1], object_fit="contain", height=480, allow_preview=True)

        # handle on-click events - generate images
        txt2img_gen_btn.click(play_audio, [], [audio])
        txt2img_gen_btn.click(fn=text_to_anime, inputs=[prompt_box, neg_prompt_box, height, width], outputs=[gallery])
        
        img2img_gen_btn.click(play_audio, [], [audio])
        img2img_gen_btn.click(fn=real_img2img_to_anime, inputs=[prompt_box, neg_prompt_box, img2img_image_box], outputs=[gallery])
        
        scribble2img_gen_btn.click(play_audio, [], [audio])
        scribble2img_gen_btn.click(fn=scribble_to_image, inputs=[prompt_box, neg_prompt_box, scribble2img_image_box, height, width], outputs=[gallery])
        
        live_scribble2img_gen_btn.click(play_audio, [], [audio])
        live_scribble2img_gen_btn.click(fn=live_scribble, inputs=[prompt_box, neg_prompt_box, live_scribble2img_image_box], outputs=[gallery])
        
        lineart_gen_btn.click(play_audio, [], [audio])
        lineart_gen_btn.click(fn=lineart, inputs=[prompt_box, neg_prompt_box, lineart_image_box], outputs=[gallery])
    
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
                    audio_gen_btn.click(voice_infer,
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
        pose_gen_btn.click(play_audio, [], [audio])
        pose_gen_btn.click(fn=pose_to_anime, inputs=[pose_prompt_box, pose_neg_prompt_box, pose_input_img], outputs=[pose_gallery])
        
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
        inpaint_image_box = gr.ImageEditor(interactive=True, type="pil", height=780, brush=gr.Brush(default_size="10"))
        placeholder_image_box = gr.ImageEditor(visible=False)
        
        # transport selected image from prototye creation to inpainting tab
        pose_gallery.select(get_select_image,  [prompt_box, neg_prompt_box], [placeholder_image_box, placeholder, placeholder])
        inpaint_btn = gr.Button(value="Inpaint")
        
        # gallery to show generated images
        inpaint_gallery = gr.Gallery(
            label="Inpainted Images", show_label=True, elem_id="gallery"
        , columns=[4], rows=[1], object_fit="contain", height=512)
        
        # handle on-click events - generate images
        inpaint_btn.click(play_audio, [], [audio])
        inpaint_btn.click(fn=inpaint, inputs=[inpaint_prompt_box, inpaint_neg_prompt_box, inpaint_image_box], outputs=[inpaint_gallery])
    
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

        with gr.Row():
            vid_neg_prompt_box = gr.Textbox(label="Negative Prompt", placeholder="Enter a negative prompt(things you don't want to include in the generated image)", lines=3)
        video_steps = gr.Slider(1, 500, label="Inference Steps", value=250)
        with gr.Row():
            num_frames = gr.Slider(10, 40, label="Number of Frames", value=10)
            fps_slider = gr.Slider(20, 50, label="FPS", value=40)

        video_gen_btn = gr.Button(value="Generate Video")
        
        with gr.Row():
            video_image = gr.Image(label="Video Image", type='pil')
            
            video1_box = gr.Video(label="Video", height=512)
            
        # handle on-click events - generate videos
        video_gen_btn.click(play_audio, [], [audio])
        video_gen_btn.click(fn=I2VGen_video, inputs=[vid_prompt_box, vid_neg_prompt_box, video_steps, num_frames, video_image, fps_slider], 
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
        avatar_gen_btn.click(play_audio, [], [audio])

        avatar_gen_btn.click(fn=chibi, inputs=[avatar_prompt_box, avatar_neg_prompt_box, avatar_ref_image, avatar_height, avatar_width], outputs=[avatar_gallery])
        
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
        remove_bg_btn.click(play_audio, [], [audio])
        remove_bg_btn.click(fn=rmbg_fn, inputs=[anime_char_image], outputs=[anime_char_remove_bg_image])
    
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
            video_release_btn = gr.Button(value="Video Generator - 20%")
            chatbot_release_btn = gr.Button(value="Chatbot - 10%")
            scribble2img_release_btn = gr.Button(value="Scribble to Image Generator - 10%")
            txt2img_release_btn = gr.Button(value="Text to Image Generator - 10%")
            img2img_release_btn = gr.Button(value="Image to Image Generator - 10%")
            lineart2img_release_btn = gr.Button(value="Lineart to Image Generator - 10%")
            pose2img_release_btn = gr.Button(value="Pose Variation Generator - 10%")
            inpaint_release_btn = gr.Button(value="Inpainting Generator - 10%")
            chibi_release_btn = gr.Button(value="Chibi Generator - 5%")
            rmbg_release_btn = gr.Button(value="Remove Background Generator - 5%")
            
            # handle on-click events - model deletion
            video_release_btn.click(fn=pipe_video.delete_model, inputs=[], outputs=[])
            chatbot_release_btn.click(fn=pipe_chatbot.delete_model, inputs=[], outputs=[])
            scribble2img_release_btn.click(fn=pipe_scribble.delete_model, inputs=[], outputs=[])
            txt2img_release_btn.click(fn=pipe_txt2img.delete_model, inputs=[], outputs=[])
            img2img_release_btn.click(fn=pipe_img2img.delete_model, inputs=[], outputs=[])
            lineart2img_release_btn.click(fn=pipe_lineart.delete_model, inputs=[], outputs=[])
            pose2img_release_btn.click(fn=pipe_pose.delete_model, inputs=[], outputs=[])
            inpaint_release_btn.click(fn=pipe_inpaint.delete_model, inputs=[], outputs=[])
            chibi_release_btn.click(fn=pipe_chibi.delete_model, inputs=[], outputs=[])
            rmbg_release_btn.click(fn=rmbg_model.delete_model, inputs=[], outputs=[])


# function that runs the Gradio interface
def run():
    iface.launch(share=True, debug=True, inline=True)

# close any opened interface
iface.close()

# run the app interface
run()

