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

def show_anns_with_boundaries_and_overlay(anns, image, output_path):
    if len(anns) == 0:
        return

    # Create a copy of the image to draw boundaries on
    image_with_boundaries_and_overlay = image.copy()
    img_height, img_width = image.shape[:2]

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    for ann in sorted_anns:
        mask = ann['segmentation']
        color = np.random.randint(0, 256, size=3)
        thickness = 2  # You can adjust the thickness of the boundary
        alpha = 0.2  # Adjust the transparency of the mask overlay (0.0 - 1.0)

        # Find contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calculate the bounding rectangle for each contour
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the contour is the edge of the image
            if w < img_width * 0.95 and h < img_height * 0.95:  # Adjust the threshold as needed
                # Draw the boundary of each mask on the image
                cv2.drawContours(image_with_boundaries_and_overlay, [contour], -1, color.tolist(), thickness)

                # Create a mask with the same shape as the image for overlay
                mask_overlay = np.zeros_like(image)
                mask_overlay[mask] = color
                
                # Blend the mask overlay onto the image with transparency
                image_with_boundaries_and_overlay = cv2.addWeighted(image_with_boundaries_and_overlay, 1, mask_overlay, alpha, 0)

    # Save the image with boundaries and overlay to the specified output path
    cv2.imwrite(output_path, cv2.cvtColor(image_with_boundaries_and_overlay, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON file with boundary coordinates and segmentation image")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--output_image", type=str, required=True, help="Path to the output segmentation image file")
    parser.add_argument("--output_image_bw", type=str, required=True, help="Path to the output segmentation image file")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model file")

    args = parser.parse_args()

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry["default"](checkpoint=args.checkpoint)
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

    show_anns_with_boundaries_and_overlay(masks, img, args.output_image)

    # Save the segmentation image to the specified output path
    cv2.imwrite(args.output_image_bw, cv2.cvtColor(segmentation_image, cv2.COLOR_RGB2BGR))

    print(f"Boundary coordinates saved to: {args.output_json}")
    print(f"Segmentation image saved to: {args.output_image}")
