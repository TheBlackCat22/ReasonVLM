from PIL import Image, ImageDraw

def apply_viser_scaffolding(img, num_lines=3):
    """Draws equidistant horizontal lines on the image."""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    for i in range(1, num_lines + 1):
        y = int(i * height / (num_lines + 1))
        draw.line([(0, y), (width, y)], fill="red", width=2)
    return img

def get_viser_prompt(prompt):
    """Wraps prompt with VISER sequential scanning instructions."""
    instruction = "The image is divided by horizontal lines. Please scan the image sequentially from top to bottom, anchoring your reasoning to these lines. "
    return instruction + prompt
