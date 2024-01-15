# -*- coding: UTF-8 -*-
import modules.scripts as scripts
import gradio as gr
import os
import modules
from modules import script_callbacks
import torch
import imageio
from diffusers import TextToVideoZeroPipeline
import numpy as np



model_id = "/your_stable diffusion pipeline"
pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")



def on_ui_tabs():


    def create_video_gradio(frames, fps, path=None):

        if path is None:
            dir = "temporal"
            os.makedirs(dir, exist_ok=True)
            path = os.path.join(dir, 'movie.mp4')

        writer = imageio.get_writer(path, format='FFMPEG', fps=fps)
        for i in frames:
            x = (i * 255).astype("uint8")
            writer.append_data(x)
        writer.close()

        return path



    def t2v(prompt, n_prompt, chunk_size, video_length, seed, fps, num_inference_steps):
        result = []
        chunk_ids = np.arange(0, video_length, chunk_size - 1)
        generator = torch.Generator(device="cuda")
        for i in range(len(chunk_ids)):
            print(f"Processing chunk {i + 1} / {len(chunk_ids)}")
            ch_start = chunk_ids[i]
            ch_end = video_length if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
            # Attach the first frame for Cross Frame Attention
            frame_ids = [0] + list(range(ch_start, ch_end))
            # Fix the seed for the temporal consistency
            generator.manual_seed(seed)
            output = pipe(prompt=prompt, video_length=len(frame_ids), generator=generator, frame_ids=frame_ids)
            result.append(output.images[1:])

        # Concatenate chunks and save
        result = np.concatenate(result)
        return create_video_gradio(result,fps)

    
    def fn_video_len_change(frames, fps):
        return gr.Textbox.update(value=str(int(frames/fps)))
    
    with gr.Blocks(analytics_enabled=False) as text2video:

        with gr.Row():
            gr.Markdown('## 🌟 A extension for automatic1111 webui text2video-zero')

        with gr.Row():
            with gr.Column():

                prompt = gr.Dropdown(choices=['A river flows slowly in the village',
                                              ''],
                                     allow_custom_value=True,
                                     label='Prompt'
                        )
                run_button = gr.Button('Run')
                with gr.Accordion('Advanced options', open=False):

  
                    video_length = gr.Slider(
                        label="Video frame rate", minimum=4, maximum=40, value=4, step=1)

                    n_prompt = gr.Textbox(
                        label="Negative prompt", value='')
                    seed = gr.Slider(label='Seed',
                                     info="-1 for random seed on each run. Otherwise, the seed will be fixed.",
                                     minimum=-1,
                                     maximum=65536,
                                     value=1234,
                                     step=1)

                    chunk_size = gr.Slider(
                        label="frames in a batch", minimum=2, maximum=16, value=8, step=1, visible=True,
                        info="The fewer frames processed simultaneously, the smaller the memory usage, but at the same time, the slower the speed."
                    )
                    fps = gr.Slider(
                        label="fps", minimum=2, maximum=16, value=2, step=1, visible=True,
                        info="frames of images per second."
                    )
                    inference_steps = gr.Slider(
                        label="inference steps", minimum=50, maximum=200, value=50, step=10, visible=True,
                        info="The larger the frame, the stronger the correlation between frames, and the longer the time consumption"
                    )
                    video_time = gr.Textbox(label='Video duration', value='2秒', interactive=False)
                    video_length.change(fn=fn_video_len_change, inputs=[video_length, fps], outputs=video_time)
                    fps.change(fn=fn_video_len_change, inputs=[video_length, fps], outputs=video_time)
                    
            with gr.Column():
                result = gr.Video(label="Generated Video")

        inputs = [
            prompt,
            n_prompt,
            chunk_size,
            video_length,
            seed,
            fps,
            inference_steps
        ]


        run_button.click(fn=t2v,
                         inputs=inputs,
                         outputs=result)

    # the third parameter is the element id on html, with a "tab_" as prefix
    return (text2video , "文生视频", "text2video"),

script_callbacks.on_ui_tabs(on_ui_tabs)


