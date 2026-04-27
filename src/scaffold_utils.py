from PIL import Image, ImageDraw, ImageFont

def apply_scaffold_coordinates(img, grid_size=(10, 10), dot_color=(255, 0, 0), text_color=(255, 0, 0)):
    """Applies centered coordinate scaffolding (-50 to 50)."""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    cols, rows = grid_size
    x_spacing, y_spacing = width / (cols + 1), height / (rows + 1)

    try:
        font_size = max(10, int(min(width, height) / 50))
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            x, y = int(c * x_spacing), int(r * y_spacing)
            draw.ellipse([x-2, y-2, x+2, y+2], fill=dot_color)
            # Reverted to top-left origin (0 to 100)
            norm_x = int((c / (cols + 1)) * 100)
            norm_y = int((r / (rows + 1)) * 100)
            draw.text((x + 5, y + 5), f"({norm_x},{norm_y})", fill=text_color, font=font)
    return img

def get_scaffold_prompt(prompt):
    """Wraps prompt with SCAFFOLD instructions."""
    instruction = "The image has a coordinate dot matrix centered at (0,0). Use these coordinates for visual grounding. "
    return instruction + prompt
