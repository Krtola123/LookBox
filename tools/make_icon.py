"""Dev-only: draw the app icon (lookbox/ui/icon.png + icon.ico). Needs Pillow.

A purple rounded tile with a light box and a soft drop shadow: "a box of looks".
Run from the repo root: python tools/make_icon.py
"""

from PIL import Image, ImageDraw, ImageFilter

S = 1024
ACCENT = (139, 61, 255, 255)  # theme accent #8b3dff


def draw() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(tile).rounded_rectangle((40, 40, S - 40, S - 40), radius=220, fill=ACCENT)
    img.alpha_composite(tile)
    # soft shadow under the box
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((300, 360, 760, 790), radius=70, fill=(40, 0, 90, 170))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(40)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((260, 290, 764, 734), radius=70, fill=(250, 248, 255, 255))
    # a "photo" inside the box: horizon + sun, in the accent colour
    d.rounded_rectangle((330, 360, 694, 664), radius=36, fill=(222, 210, 255, 255))
    d.polygon([(330, 664), (450, 520), (540, 600), (610, 540), (694, 664)], fill=ACCENT)
    d.ellipse((590, 400, 650, 460), fill=ACCENT)
    return img


if __name__ == "__main__":
    icon = draw()
    icon.resize((256, 256), Image.LANCZOS).save("lookbox/ui/icon.png")
    icon.save("lookbox/ui/icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote lookbox/ui/icon.png and icon.ico")
