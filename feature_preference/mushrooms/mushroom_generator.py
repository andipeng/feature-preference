from PIL import Image

def merge_images(image1_path, image2_path, output_path):
    # Open images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    # Get dimensions of the second image
    width_before, height_before = image2.size
    image2 = image2.crop((0, int((height_before//3)*2), width_before, height_before))

    # Get dimensions of the first image
    width1, height1 = image1.size

    # Get new dimensions of the second image after crop
    width2, height2 = image2.size

    # Determine the size of the new image
    new_width = max(width1, width2)
    new_height = height1 + height2

    # resize image2
    image2 = image2.resize((new_width,height2))

    # Create a new image with the determined size
    new_image = Image.new('RGB', (new_width, new_height))

    # Paste the first image on the up
    new_image.paste(image1, (0, 0))

    # Paste the second image on the bottom
    new_image.paste(image2, (0, height1))

    # Save the merged image
    new_image.save(output_path)


if __name__ == "__main__":
    # Replace 'image1.jpg', 'image2.jpg', and 'output.jpg' with image file paths
    merge_images('feature_map.jpg', 'heavy.jpg', 'output.jpg')

