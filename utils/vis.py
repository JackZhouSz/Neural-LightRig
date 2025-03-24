from PIL import Image


def concat_images_vertically(images, mode='RGBA'):
    # Get the maximum width and the total height for the new image
    max_width = max(image.width for image in images)
    total_height = sum(image.height for image in images)
    
    # Create a new blank image with max width and total height
    new_image = Image.new(mode, (max_width, total_height))
    
    # Initialize the y_offset to stack the images from top to bottom
    y_offset = 0
    for image in images:
        new_image.paste(image, (0, y_offset))
        y_offset += image.height  # Move the y_offset for the next image
    
    # Save the final concatenated image
    return new_image


def replace_bg_preserving_alpha(img_in: Image, bg: int) -> Image:
    assert img_in.mode == "RGBA", "Input image must be RGBA"
    assert 0 <= bg <= 255, "Background color must be in [0, 255]"
    background = Image.new("RGB", img_in.size, (bg, bg, bg))
    alpha = img_in.getchannel("A")
    img_rgb = Image.composite(img_in, background, alpha)
    img_rgba = img_rgb.convert("RGBA")
    img_rgba.putalpha(alpha)
    return img_rgba
