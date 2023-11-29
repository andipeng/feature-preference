from mushroom_generator import merge_images
from background_filling import merge_images_with_background
from resizing import resize_and_crop
from PIL import Image

def generate_image(shape, color, texture, smell, size, weight):
    # filename for the non-empty charactor
    filename_parts = [part.lower() for part in [shape, color, texture, smell, size, weight] if part]
    # if empty
    if not filename_parts:
        print("\n No input provided. Please enter at least one characteristic.")
        return

    # check for inputs and any empty input
    if shape and not any([color, texture, smell, size, weight]):
        filename = f"{shape.lower()}.jpg"

    elif color and not any([texture, smell, size, weight]):
        filename = f"{color.lower()}_{shape.lower()}.jpg"

    elif texture and not any([color, smell, size, weight]):
        print("\n Please provide more information")
        return

    elif smell and not any([color, texture, size, weight]):
        print("\n Please provide more information")
        return

    elif size and not any([color, texture, smell, weight]):
        print("\n Please provide more information")
        return

    elif weight and not any([color, texture, size, smell]):
        print("\n Please provide more information")
        return

    else:
        # all other inputs
        initialname_parts = [color, shape]
        name = "_".join(initialname_parts).lower()
        filename = f"{name}.jpg"

        if size:
            input_path = filename
            filename = "_".join(filename_parts).lower() + ".jpg"
            resize_and_crop(input_path, filename, size)

        if texture:
            image1_path = filename
            image2_path = f"{texture.lower()}.png"
            filename = "_".join(filename_parts).lower() + ".jpg"
            merge_images_with_background(image1_path, image2_path, filename)

        if smell:
            image1_path = filename
            image2_path = f"{smell.lower()}.png"
            filename = "_".join(filename_parts).lower() + ".jpg"
            merge_images_with_background(image1_path, image2_path, filename)


        if weight:
            image1_path = filename
            image2_path = f"{weight.lower()}.jpg"
            filename = "_".join(filename_parts).lower() + ".jpg"
            merge_images(image1_path, image2_path, filename)
            
    try:
        # Open the image
        image = Image.open(filename)
        image.show()
    except FileNotFoundError:
        print(f"Image not found for {', '.join(filename_parts)}.")



if __name__ == "__main__":
    # Get input
    shape = input("Enter shape (umbrella, round, cylinder): ")
    color = input("Enter color (green, red, blue): ")
    texture = input("Enter texture (dot, smooth, net): ")
    smell = input("Enter smell (good, neutral, stinky): ")
    size = input("Enter size (small, medium, large): ")
    weight = input("Enter weight (light, average, heavy): ")

    # Generate and display image
    generate_image(shape, color, texture, smell, size, weight)

