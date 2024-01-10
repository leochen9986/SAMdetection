import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch


def show_anns_with_boundaries_and_overlay(anns, image, output_path):
    if len(anns) == 0:
        return

    # Create a copy of the image to draw boundaries on
    image_with_boundaries_and_overlay = image.copy()

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    for ann in sorted_anns:
        mask = ann['segmentation']
        color = np.random.randint(0, 256, size=3)
        thickness = 2  # You can adjust the thickness of the boundary
        alpha = 0.1  # Adjust the transparency of the mask overlay (0.0 - 1.0)

        # Find contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the boundary of each mask on the image
        cv2.drawContours(image_with_boundaries_and_overlay, contours, -1, color.tolist(), thickness)

        # Create a mask with the same shape as the image
        mask_overlay = np.zeros_like(image)
        mask_overlay[mask] = color

        # Blend the mask overlay onto the image with transparency
        image_with_boundaries_and_overlay = cv2.addWeighted(image_with_boundaries_and_overlay, 1, mask_overlay, alpha, 0)

    # Save the image with boundaries and overlay to the specified output path
    cv2.imwrite(output_path, cv2.cvtColor(image_with_boundaries_and_overlay, cv2.COLOR_RGB2BGR))

    # Display the image with boundaries and overlay using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_boundaries_and_overlay)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask overlay and boundary drawing on an image")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output image file")

    args = parser.parse_args()

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(img)

    # Specify the output image path where the annotated image will be saved
    output_image_path = args.output

    # Call the function to show the annotated image with boundaries and overlay and export it
    show_anns_with_boundaries_and_overlay(masks, img, output_image_path)

    print(f"Annotated image with boundaries and overlay saved to: {output_image_path}")
