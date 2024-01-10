# SAMdetection

## Description
This script overlays masks with transparency on an input image and draws boundaries around each mask/segmentation using the Segment Anything Model (SAM). SAM produces high-quality object masks from input prompts and has strong zero-shot performance on segmentation tasks.

## Prerequisites
- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- PyTorch (>=1.7)
- TorchVision (>=0.8)

## SAM Installation
1. Install PyTorch and TorchVision with CUDA support as recommended by the SAM library. Follow the instructions [here](https://github.com/facebookresearch/segment-anything#installation) to install PyTorch and TorchVision.

2. Install Segment Anything library using pip:

   ```bash
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```
Alternatively, clone the SAM repository locally and install it:
    ```bash
    git clone git@github.com:facebookresearch/segment-anything.git
    cd segment-anything
    pip install -e .
    ```

Install additional dependencies for mask post-processing, saving masks in COCO format, example notebooks, and exporting the model in ONNX format:
   ```bash
   pip install opencv-python pycocotools matplotlib onnxruntime onnx
   ```

## SAM Model Checkpoint
Three model versions of SAM are available with different backbone sizes:

- ViT-H SAM model (default)
- ViT-L SAM model
- ViT-B SAM model

**Step 1:** Download the ViT-H SAM model checkpoint from the provided link in the [SAM README](https://github.com/facebookresearch/segment-anything#model-checkpoints).

**Step 2:** Place the downloaded checkpoint file in the project directory.

**Step 3:** When running the script, specify the path to the ViT-H checkpoint using the `--checkpoint` argument.


## Usage
Run the script from the command prompt, specifying the input image and output image file paths as arguments:
   ```bash
   python mask_overlay_and_boundary.py --input input_image.jpg --output output_image.jpg
   ```


Replace input_image.jpg with the path to your input image and output_image.jpg with the desired output image path

## Output
The script will save the annotated image with boundaries and overlay to the specified output image file path.

