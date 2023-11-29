from PIL import Image

def resize_and_crop(input_path, output_path, size):
    # Open the original image
    original_image = Image.open(input_path)

    # Calculate the target size based on the original dimensions
    original_width, original_height = original_image.size

    if size.lower() == 'small':
        target_width, target_height = original_width // 2, original_height // 2
        # Calculate crop coordinates to get the center of the resized image
        left = 0
        top = 0
        right = 1500
        bottom = 1500
        x = 750
        y = 750
    elif size.lower() == 'medium':
        target_width, target_height = original_width, original_height
        # Calculate crop coordinates to get the center of the resized image
        left = (target_width - original_width) // 2
        top = (target_height - original_height) // 2
        right = left + original_width
        bottom = top + original_height
        x = 450
        y = 450
    elif size.lower() == 'large':
        target_width, target_height = int(original_width * 1.5), int(original_height * 1.5)
        # Calculate crop coordinates to get the center of the resized image
        left = (target_width - original_width) // 2
        top = (target_height - original_height) // 2
        right = left + original_width
        bottom = top + original_height
        x = 300
        y = 300

    # Resize the image
    resized_image = original_image.resize((target_width, target_height))


    # Crop the middle of the resized image
    cropped_image = resized_image.crop((left, top, right, bottom))

    # Create a new image with the same size as the original, filled with white background
    new_image = Image.new("RGB", (original_width, original_height), "white")
    cropped_image = cropped_image.convert("RGB")
    new_image.paste(cropped_image, (x,y))

    # Save the cropped image
    new_image.save(output_path, "PNG")

if __name__ == "__main__":
    # Replace 'input_image.jpg' and 'output_image.jpg' with your image file paths
    input_image_path = 'blue_umbrella.jpg'
    output_image_path = 'output_image3.jpg'

    # Specify the target size: 'small', 'medium', or 'large'
    target_size = 'small'

    # Call the function to resize and crop the image
    resize_and_crop(input_image_path, output_image_path, target_size)
