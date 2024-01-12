import argparse
import cv2
import numpy as np
import json
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch

def get_boundary_coordinates(anns):
    boundary_coordinates = []
    id_counter = 1

    for ann in anns:
        mask = ann['segmentation']

        # Find contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract coordinates of the boundary
        boundary_coords = [contour.squeeze(1).tolist() for contour in contours]

        # Add boundary coordinates to the list with ID
        boundary_coordinates.append({
            "id": id_counter,
            "outline": [{"x": int(coord[0]), "y": int(coord[1])} for coord in boundary_coords[0]]
        })

        id_counter += 1

    return boundary_coordinates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON file with boundary coordinates and segmentation image")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--output_image", type=str, required=True, help="Path to the output segmentation image file")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file")

    args = parser.parse_args()

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry["default"](checkpoint=r"C:\Users\Laptop\Downloads\sam_vit_h_4b8939.pth")
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(img)

    # Extract boundary coordinates
    boundary_coordinates = get_boundary_coordinates(masks)

    # Create JSON output format
    json_output = {
        "objects": boundary_coordinates
    }

    # Save the JSON coordinates to the specified output JSON file
    with open(args.output_json, 'w') as json_file:
        json.dump(json_output, json_file, indent=2)

    # Generate the black and white segmentation image
    segmentation_image = np.zeros_like(img)
    for mask in masks:
        segmentation_image[mask['segmentation']] = [255, 255, 255]

    # Save the segmentation image to the specified output path
    cv2.imwrite(args.output_image, cv2.cvtColor(segmentation_image, cv2.COLOR_RGB2BGR))

    print(f"Boundary coordinates saved to: {args.output_json}")
    print(f"Segmentation image saved to: {args.output_image}")
