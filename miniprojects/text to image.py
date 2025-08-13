from PIL import Image, ImageDraw, ImageFont

def text_to_image(text, font_path="arial.ttf", font_size=40, output_path="text_image.png"):
    font = ImageFont.truetype(font_path, font_size)

    temp_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.textbbox((0, 0), text, font=font)

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    image = Image.new("RGB", (width + 20, height + 20), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, fill="black", font=font)

    image.save(output_path)

# Example
text_to_image("High-res text\nwith big font!", font_size=60)
