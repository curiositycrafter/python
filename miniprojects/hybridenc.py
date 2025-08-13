import random
import string
from cryptography.fernet import Fernet as f
from PIL import Image, ImageDraw, ImageFont

chars =  string.punctuation +' '+ string.digits+' '+ string.ascii_letters+' '
chars = list(chars)
keyy = chars.copy()

random.shuffle(keyy)

plain_text = input("Enter a message to encrypt: ")
cipher_text = ""

for letter in plain_text:
    index = chars.index(letter)
    cipher_text += keyy[index]

key = f.generate_key()
ins=f(key)
msg = cipher_text.encode()   
msgv2 = ins.encrypt(msg)


def text_to_image(text, font_path="arial.ttf", font_size=40, output_path="enc.png"):
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

text_to_image(msgv2, font_size=60)


#steg



cipher_text=ins.decrypt(msgv2).decode() 

for letter in cipher_text:
    index = keyy.index(letter)
    plain_text += chars[index]

print(f"original message : {plain_text}")
