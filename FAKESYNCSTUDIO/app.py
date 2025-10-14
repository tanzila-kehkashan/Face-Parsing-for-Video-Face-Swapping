import os
import cv2
import glob
import time
import torch
import shutil
import argparse
import platform
import datetime
import subprocess
import insightface
import onnxruntime
import numpy as np
import gradio as gr
import threading
import queue
from tqdm import tqdm
import concurrent.futures
from moviepy.editor import VideoFileClip

from face_swapper import Inswapper, paste_to_whole
from face_analyser import detect_conditions, get_analysed_data, swap_options_list
from face_parsing import init_parsing_model, get_parsed_mask, mask_regions, mask_regions_to_list
from face_enhancer import get_available_enhancer_names, load_face_enhancer_model
from utils import trim_video, StreamerThread, ProcessBar, open_directory, split_list_by_lengths, merge_img_sequence_from_ref, create_image_grid

## ------------------------------ USER ARGS ------------------------------

parser = argparse.ArgumentParser(description="AI Face Swap - Professional Quality")
parser.add_argument("--out_dir", help="Default Output directory", default=os.getcwd())
parser.add_argument("--batch_size", help="Gpu batch size", default=32)
parser.add_argument("--cuda", action="store_true", help="Enable cuda", default=False)
parser.add_argument("--colab", action="store_true", help="Enable colab mode", default=False)
user_args = parser.parse_args()

## ------------------------------ OPTIMIZED DEFAULTS ------------------------------

USE_COLAB = user_args.colab
USE_CUDA = user_args.cuda
DEF_OUTPUT_PATH = user_args.out_dir
BATCH_SIZE = int(user_args.batch_size)
WORKSPACE = None
OUTPUT_FILE = None
CURRENT_FRAME = None
STREAMER = None
DETECT_CONDITION = "best detection"
DETECT_SIZE = 640
DETECT_THRESH = 0.6
NUM_OF_SRC_SPECIFIC = 3  # Reduced from 10 to 3

# Optimal settings - always used
FACE_ENHANCER_DEFAULT = "CodeFormer"
FACE_PARSING_ENABLED = True

# Optimal face parsing settings
OPTIMAL_MASK_INCLUDE = [
    "Skin",
    "R-Eyebrow",
    "L-Eyebrow",
    "L-Eye",
    "R-Eye",
    "Nose",
    "Mouth",
    "L-Lip",
    "U-Lip"
]

# Optimal processing settings
OPTIMAL_SETTINGS = {
    'enable_face_parser': True,
    'mask_includes': OPTIMAL_MASK_INCLUDE,
    'mask_soft_kernel': 17,
    'mask_soft_iterations': 10,
    'blur_amount': 0.1,
    'erode_amount': 0.15,
    'face_scale': 1.0,
    'enable_laplacian_blend': True,
    'crop_top': 0,
    'crop_bott': 511,
    'crop_left': 0,
    'crop_right': 511,
    'swap_option': 'All Face',
    'age': 25,
    'distance_slider': 0.6,
    'keep_output_sequence': False
}

FACE_SWAPPER = None
FACE_ANALYSER = None
FACE_ENHANCER = None
FACE_PARSER = None

# Simple swap options for users
SIMPLE_SWAP_OPTIONS = [
    "All Face",
    "Biggest", 
    "All Male",
    "All Female",
    "Specific Face"
]

## ------------------------------ SET EXECUTION PROVIDER ------------------------------

PROVIDER = ["CPUExecutionProvider"]

if USE_CUDA:
    available_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        print("\n********** Running on CUDA **********\n")
        PROVIDER = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        USE_CUDA = False
        print("\n********** CUDA unavailable running on CPU **********\n")
else:
    USE_CUDA = False
    print("\n********** Running on CPU **********\n")

device = "cuda" if USE_CUDA else "cpu"
EMPTY_CACHE = lambda: torch.cuda.empty_cache() if device == "cuda" else None

## ------------------------------ LOAD MODELS ------------------------------

def load_face_analyser_model(name="buffalo_l"):
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name=name, providers=PROVIDER)
        FACE_ANALYSER.prepare(
            ctx_id=0, det_size=(DETECT_SIZE, DETECT_SIZE), det_thresh=DETECT_THRESH
        )

def load_face_swapper_model(path="./assets/pretrained_models/inswapper_128.onnx"):
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        batch = int(BATCH_SIZE) if device == "cuda" else 1
        FACE_SWAPPER = Inswapper(model_file=path, batch_size=batch, providers=PROVIDER)

def load_face_parser_model(path="./assets/pretrained_models/79999_iter.pth"):
    global FACE_PARSER
    if FACE_PARSER is None:
        FACE_PARSER = init_parsing_model(path, device=device)

# Load models at startup
load_face_analyser_model()
load_face_swapper_model()

## ------------------------------ MAIN PROCESS FUNCTION ------------------------------

def process(
    input_type,
    image_path,
    video_path,
    directory_path,
    source_path,
    output_path,
    output_name,
    swap_condition="All Face",
    *specifics,
):
    """
    Simplified process function with optimal defaults
    All settings are automatically optimized for best results
    """
    global WORKSPACE
    global OUTPUT_FILE
    global PREVIEW
    WORKSPACE, OUTPUT_FILE, PREVIEW = None, None, None

    ## ------------------------------ UI UPDATE FUNCTIONS ------------------------------
    def ui_before():
        return (
            gr.update(visible=True, value=PREVIEW),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(visible=False),
        )

    def ui_after():
        return (
            gr.update(visible=True, value=PREVIEW),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(visible=False),
        )

    def ui_after_vid():
        return (
            gr.update(visible=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=OUTPUT_FILE, visible=True),
        )

    start_time = time.time()
    total_exec_time = lambda start_time: divmod(time.time() - start_time, 60)
    get_finsh_text = lambda start_time: f"✔️ Completed in {int(total_exec_time(start_time)[0])} min {int(total_exec_time(start_time)[1])} sec."

    ## ------------------------------ LOAD MODELS WITH OPTIMAL SETTINGS ------------------------------
    
    yield "### \n 🌀 Loading face analyser model...", *ui_before()
    load_face_analyser_model()

    yield "### \n ⚙️ Loading face swapper model...", *ui_before()
    load_face_swapper_model()

    # Always load CodeFormer enhancer
    yield f"### \n 💡 Loading {FACE_ENHANCER_DEFAULT} enhancer...", *ui_before()
    FACE_ENHANCER = load_face_enhancer_model(name=FACE_ENHANCER_DEFAULT, device=device)

    # Always load face parsing for optimal results
    yield "### \n 📀 Loading face parsing model...", *ui_before()
    load_face_parser_model()

    # Use optimal settings
    settings = OPTIMAL_SETTINGS.copy()
    settings['swap_option'] = swap_condition
    
    # Convert specifics for specific face functionality  
    specifics = list(specifics)
    half = len(specifics) // 2
    sources = specifics[:half]
    specifics = specifics[half:]
    
    includes = mask_regions_to_list(settings['mask_includes'])
    crop_mask = (settings['crop_top'], 511-settings['crop_bott'], 
                 settings['crop_left'], 511-settings['crop_right'])

    def swap_process(image_sequence):
        ## ------------------------------ FACE ANALYSIS ------------------------------
        yield "### \n 🧿 Analysing face data...", *ui_before()
        
        if swap_condition != "Specific Face":
            source_data = source_path, settings['age']
        else:
            source_data = ((sources, specifics), settings['distance_slider'])
            
        analysed_targets, analysed_sources, whole_frame_list, num_faces_per_frame = get_analysed_data(
            FACE_ANALYSER,
            image_sequence,
            source_data,
            swap_condition=swap_condition,
            detect_condition=DETECT_CONDITION,
            scale=settings['face_scale']
        )

        ## ------------------------------ FACE SWAPPING ------------------------------
        yield "### \n 🧶 Generating faces...", *ui_before()
        preds = []
        matrs = []
        count = 0
        global PREVIEW
        
        for batch_pred, batch_matr in FACE_SWAPPER.batch_forward(whole_frame_list, analysed_targets, analysed_sources):
            preds.extend(batch_pred)
            matrs.extend(batch_matr)
            EMPTY_CACHE()
            count += 1

            if USE_CUDA:
                image_grid = create_image_grid(batch_pred, size=128)
                PREVIEW = image_grid[:, :, ::-1]
                yield f"### \n 🧩 Generating face Batch {count}", *ui_before()

        ## ------------------------------ FACE ENHANCEMENT (ALWAYS ENABLED) ------------------------------
        generated_len = len(preds)
        yield f"### \n 🎲 Enhancing faces with {FACE_ENHANCER_DEFAULT}...", *ui_before()
        
        for idx, pred in tqdm(enumerate(preds), total=generated_len, desc=f"Enhancing with {FACE_ENHANCER_DEFAULT}"):
            enhancer_model, enhancer_model_runner = FACE_ENHANCER
            pred = enhancer_model_runner(pred, enhancer_model)
            preds[idx] = cv2.resize(pred, (512,512))
        EMPTY_CACHE()

        ## ------------------------------ FACE PARSING (ALWAYS ENABLED) ------------------------------
        yield "### \n 🎨 Creating optimal face masks...", *ui_before()
        masks = []
        count = 0
        
        for batch_mask in get_parsed_mask(
            FACE_PARSER, 
            preds, 
            classes=includes, 
            device=device, 
            batch_size=BATCH_SIZE, 
            softness=int(settings['mask_soft_iterations'])
        ):
            masks.append(batch_mask)
            EMPTY_CACHE()
            count += 1

            if len(batch_mask) > 1:
                image_grid = create_image_grid(batch_mask, size=128)
                PREVIEW = image_grid[:, :, ::-1]
                yield f"### \n 🪙 Processing face masks Batch {count}", *ui_before()
                
        masks = np.concatenate(masks, axis=0) if len(masks) >= 1 else masks

        ## ------------------------------ FINAL PROCESSING ------------------------------
        split_preds = split_list_by_lengths(preds, num_faces_per_frame)
        del preds
        split_matrs = split_list_by_lengths(matrs, num_faces_per_frame)
        del matrs
        split_masks = split_list_by_lengths(masks, num_faces_per_frame)
        del masks

        ## ------------------------------ PASTE-BACK WITH OPTIMAL SETTINGS ------------------------------
        yield "### \n 🧿 Applying final composition...", *ui_before()
        
        def post_process(frame_idx, frame_img, split_preds, split_matrs, split_masks):
            whole_img_path = frame_img
            whole_img = cv2.imread(whole_img_path)
            blend_method = 'laplacian' if settings['enable_laplacian_blend'] else 'linear'
            
            for p, m, mask in zip(split_preds[frame_idx], split_matrs[frame_idx], split_masks[frame_idx]):
                p = cv2.resize(p, (512,512))
                mask = cv2.resize(mask, (512,512)) if mask is not None else None
                m /= 0.25
                whole_img = paste_to_whole(
                    p, whole_img, m, 
                    mask=mask, 
                    crop_mask=crop_mask, 
                    blend_method=blend_method, 
                    blur_amount=settings['blur_amount'], 
                    erode_amount=settings['erode_amount']
                )
            cv2.imwrite(whole_img_path, whole_img)

        def concurrent_post_process(image_sequence, *args):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for idx, frame_img in enumerate(image_sequence):
                    future = executor.submit(post_process, idx, frame_img, *args)
                    futures.append(future)

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Final processing"):
                    result = future.result()

        concurrent_post_process(image_sequence, split_preds, split_matrs, split_masks)

    ## ------------------------------ PROCESS DIFFERENT INPUT TYPES ------------------------------
    
    if input_type == "Image":
        target = cv2.imread(image_path)
        output_file = os.path.join(output_path, output_name + ".png")
        cv2.imwrite(output_file, target)

        for info_update in swap_process([output_file]):
            yield info_update

        OUTPUT_FILE = output_file
        WORKSPACE = output_path
        PREVIEW = cv2.imread(output_file)[:, :, ::-1]
        yield get_finsh_text(start_time), *ui_after()

    elif input_type == "Video":
        temp_path = os.path.join(output_path, output_name, "sequence")
        os.makedirs(temp_path, exist_ok=True)

        yield "### \n ⌛ Extracting video frames...", *ui_before()
        image_sequence = []
        cap = cv2.VideoCapture(video_path)
        curr_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_path = os.path.join(temp_path, f"frame_{curr_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            image_sequence.append(frame_path)
            curr_idx += 1
        cap.release()

        for info_update in swap_process(image_sequence):
            yield info_update

        yield "### \n ⌛ Creating final video...", *ui_before()
        output_video_path = os.path.join(output_path, output_name + ".mp4")
        merge_img_sequence_from_ref(video_path, image_sequence, output_video_path)

        if os.path.exists(temp_path) and not settings['keep_output_sequence']:
            yield "### \n ⌛ Cleaning up temporary files...", *ui_before()
            shutil.rmtree(temp_path)

        WORKSPACE = output_path
        OUTPUT_FILE = output_video_path
        yield get_finsh_text(start_time), *ui_after_vid()

    elif input_type == "Directory":
        extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "ico", "webp"]
        temp_path = os.path.join(output_path, output_name)
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)

        file_paths = []
        for file_path in glob.glob(os.path.join(directory_path, "*")):
            if any(file_path.lower().endswith(ext) for ext in extensions):
                img = cv2.imread(file_path)
                new_file_path = os.path.join(temp_path, os.path.basename(file_path))
                cv2.imwrite(new_file_path, img)
                file_paths.append(new_file_path)

        for info_update in swap_process(file_paths):
            yield info_update

        PREVIEW = cv2.imread(file_paths[-1])[:, :, ::-1]
        WORKSPACE = temp_path
        OUTPUT_FILE = file_paths[-1]
        yield get_finsh_text(start_time), *ui_after()

## ------------------------------ API FUNCTION ------------------------------

def process_api(
    source_image_path: str,
    target_video_path: str,
    output_directory: str,
    output_name: str = "result",
    swap_condition: str = "All Face",
    progress_callback=None
):
    """
    Simplified API with optimal settings built-in
    
    Args:
        source_image_path: Path to source face image
        target_video_path: Path to target video  
        output_directory: Output directory
        output_name: Output filename (without extension)
        swap_condition: Which faces to swap ("All Face", "Biggest", etc.)
        progress_callback: Optional callback function for progress updates
    
    Returns:
        tuple: (success: bool, output_path: str or error_message: str)
    """
    
    try:
        def log_progress(message):
            if progress_callback:
                progress_callback(message)
            print(f"API Process: {message}")
        
        log_progress("Starting face swap with optimal settings...")
        
        # Simplified arguments - no complex settings needed
        process_args = [
            "Video",  # input_type
            None,     # image_path
            target_video_path,  # video_path
            None,     # directory_path
            source_image_path,  # source_path
            output_directory,   # output_path
            output_name,        # output_name
            swap_condition,     # swap_condition (simplified)
        ]
        
        # Add empty specifics for specific face functionality
        specifics = [None] * 6  # 3 source + 3 target
        process_args.extend(specifics)
        
        log_progress("Processing with optimal AI models...")
        
        # Call the simplified process function
        process_generator = process(*process_args)
        
        # Process the generator and capture messages
        for result in process_generator:
            if isinstance(result, tuple) and len(result) >= 1:
                message = result[0]
                if isinstance(message, str):
                    log_progress(message)
            elif isinstance(result, str):
                log_progress(result)
        
        # Check if output file was created
        expected_output = os.path.join(output_directory, f"{output_name}.mp4")
        if os.path.exists(expected_output) and os.path.getsize(expected_output) > 0:
            log_progress(f"Processing completed successfully: {expected_output}")
            return True, expected_output
        else:
            # Try to find any .mp4 file in the output directory
            for file in os.listdir(output_directory):
                if file.endswith('.mp4'):
                    full_path = os.path.join(output_directory, file)
                    if os.path.getsize(full_path) > 0:
                        log_progress(f"Processing completed: {full_path}")
                        return True, full_path
            
            log_progress("Processing completed but no output file found")
            return False, "No output file was created"
    
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        if progress_callback:
            progress_callback(error_msg)
        print(f"API Process Error: {error_msg}")
        return False, error_msg

## ------------------------------ GRADIO FUNCTIONS ------------------------------

def update_radio(value):
    if value == "Image":
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif value == "Video":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    elif value == "Directory":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )

def swap_option_changed(value):
    if value == "Specific Face":
        return gr.update(visible=True)
    return gr.update(visible=False)

def stop_running():
    global STREAMER
    if hasattr(STREAMER, "stop"):
        STREAMER.stop()
        STREAMER = None
    return "Cancelled"

## ------------------------------ SIMPLIFIED GRADIO INTERFACE ------------------------------

css = """
footer{display:none !important}
.gradio-container {max-width: 1200px !important}
"""

with gr.Blocks(css=css, title="AI Face Swap - Professional Quality") as interface:
    gr.Markdown("# 🎭 **AI Face Swap** - Professional Quality Results")
    gr.Markdown("### Simply upload your images/video and get the best results with our optimized AI models")
    
    with gr.Row():
        # Left Column - Inputs and Settings
        with gr.Column(scale=0.4):
            with gr.Tab("📤 **Source & Target**"):
                # Source image
                source_image_input = gr.Image(
                    label="Source Face (The face you want to use)", 
                    type="filepath", 
                    interactive=True,
                    height=200
                )
                
                # Swap condition
                swap_option = gr.Radio(
                    SIMPLE_SWAP_OPTIONS,
                    label="Which faces to swap?",
                    value="All Face",
                    interactive=True,
                    info="Choose which faces in the target should be replaced"
                )
                
                # Specific face section (only shown when needed)
                with gr.Group(visible=False) as specific_face:
                    gr.Markdown("**For Specific Face:** Upload examples of the exact person to target")
                    with gr.Row():
                        for i in range(NUM_OF_SRC_SPECIFIC):
                            idx = i + 1
                            exec(f"""
with gr.Column():
    src{idx} = gr.Image(interactive=True, type='numpy', label=f'Source {idx}', height=120)
    trg{idx} = gr.Image(interactive=True, type='numpy', label=f'Target {idx}', height=120)
""")
                
                # Input type
                input_type = gr.Radio(
                    ["Image", "Video", "Directory"],
                    label="What do you want to process?",
                    value="Image",
                )

                # Target inputs
                with gr.Group(visible=True) as input_image_group:
                    image_input = gr.Image(
                        label="Target Image (Where faces will be swapped)", 
                        interactive=True, 
                        type="filepath",
                        height=200
                    )

                with gr.Group(visible=False) as input_video_group:
                    video_input = gr.Video(
                        label="Target Video (Where faces will be swapped)", 
                        interactive=True,
                        height=200
                    )

                with gr.Group(visible=False) as input_directory_group:
                    direc_input = gr.Text(
                        label="Directory Path (Folder containing images)", 
                        interactive=True,
                        placeholder="C:\\path\\to\\your\\images"
                    )

            with gr.Tab("💾 **Output Settings**"):
                output_directory = gr.Text(
                    label="Output Directory",
                    value=DEF_OUTPUT_PATH,
                    interactive=True,
                    info="Where to save the results"
                )
                output_name = gr.Text(
                    label="Output Name", 
                    value="FaceSwap_Result", 
                    interactive=True,
                    info="Name for your output file"
                )
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ **Optimization Features (Always Enabled)**")
                gr.Markdown(f"""
                ✅ **{FACE_ENHANCER_DEFAULT} Enhancement** - AI face restoration  
                ✅ **Face Parsing** - Precise facial region detection  
                ✅ **Laplacian Blending** - Seamless face integration  
                ✅ **Optimal Mask Processing** - Best quality blending  
                """)

        # Right Column - Processing and Results
        with gr.Column(scale=0.6):
            info = gr.Markdown(value="### 👆 Upload your source face and target image/video, then click **Swap**!")

            # Main action buttons
            with gr.Row():
                swap_button = gr.Button(
                    "🎭 **Swap Faces**", 
                    variant="primary", 
                    size="lg",
                    scale=3
                )
                cancel_button = gr.Button(
                    "⏹️ Cancel", 
                    variant="secondary",
                    scale=1
                )

            # Results
            preview_image = gr.Image(
                label="Result Preview", 
                interactive=False,
                height=400
            )
            preview_video = gr.Video(
                label="Result Video", 
                interactive=False, 
                visible=False,
                height=400
            )

            # Output buttons
            with gr.Row():
                output_directory_button = gr.Button(
                    "📁 Open Output Folder", 
                    interactive=False, 
                    visible=False
                )
                output_video_button = gr.Button(
                    "🎬 Play Result Video", 
                    interactive=False, 
                    visible=False
                )

            # Quality info
            with gr.Accordion("ℹ️ **About the Quality Settings**", open=False):
                gr.Markdown("""
                This application uses **optimized settings** for the best results:
                
                **🤖 AI Models Used:**
                - **InsightFace** - Professional face detection and analysis
                - **Inswapper** - State-of-the-art face swapping
                - **CodeFormer** - Advanced face restoration and enhancement
                - **BiSeNet** - Precise face segmentation for perfect blending
                
                **🎯 Automatic Optimizations:**
                - Intelligent face region masking
                - Soft edge blending for natural results  
                - Laplacian pyramid blending
                - Automatic quality enhancement
                - Perfect color matching
                
                **No manual tuning needed** - just upload and swap!
                """)

    ## ------------------------------ EVENT HANDLERS ------------------------------

    # Wire up event handlers
    input_type.change(
        update_radio,
        inputs=[input_type],
        outputs=[input_image_group, input_video_group, input_directory_group],
    )
    
    swap_option.change(
        swap_option_changed,
        inputs=[swap_option],
        outputs=[specific_face],
    )

    # Generate specific face inputs dynamically
    src_specific_inputs = []
    gen_variable_txt = ",".join([f"src{i+1}" for i in range(NUM_OF_SRC_SPECIFIC)] + [f"trg{i+1}" for i in range(NUM_OF_SRC_SPECIFIC)])
    exec(f"src_specific_inputs = ({gen_variable_txt})")

    # Simplified swap inputs
    swap_inputs = [
        input_type,
        image_input,
        video_input,
        direc_input,
        source_image_input,
        output_directory,
        output_name,
        swap_option,
        *src_specific_inputs,
    ]

    swap_outputs = [
        info,
        preview_image,
        output_directory_button,
        output_video_button,
        preview_video,
    ]

    # Main swap event
    swap_event = swap_button.click(
        fn=process, 
        inputs=swap_inputs, 
        outputs=swap_outputs, 
        show_progress=True
    )
    
    # Cancel event
    cancel_button.click(
        fn=stop_running,
        inputs=None,
        outputs=[info],
        cancels=[swap_event],
        show_progress=True,
    )
    
    # Output folder buttons
    output_directory_button.click(
        lambda: open_directory(path=WORKSPACE), 
        inputs=None, 
        outputs=None
    )
    output_video_button.click(
        lambda: open_directory(path=OUTPUT_FILE), 
        inputs=None, 
        outputs=None
    )

if __name__ == "__main__":
    if USE_COLAB:
        print("🚀 Running in Colab mode - Optimized for best results!")
    else:
        print("🚀 Starting Face Swap Application - Professional Quality AI")
        print("✨ All settings optimized for best results automatically!")
        
    interface.queue().launch(share=USE_COLAB, max_threads=10)