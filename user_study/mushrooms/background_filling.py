from PIL import Image

def merge_images_with_background(background_path, foreground_path, output_path):
    # Open images
    background = Image.open(background_path)
    foreground = Image.open(foreground_path)

    # Resize the foreground image to fit the background
    foreground = foreground.resize(background.size, Image.LANCZOS)

    # Composite the images
    merged_image = Image.alpha_composite(background.convert('RGBA'), foreground.convert('RGBA'))

    # Save the merged image
    merged_image.save(output_path, "PNG")

if __name__ == "__main__":
    # Replace 'background.jpg', 'foreground.jpg', and 'output.png' with image file paths
    merge_images_with_background('feature_map.jpg', 'shape_umbrella.png', 'output2.jpg')
