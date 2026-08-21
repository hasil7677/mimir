"""Generates the GitHub social preview image (1280x640) for Mimir.
Not part of the site's runtime, just the source for social-preview.png.
Run: python make_social_preview.py
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1280, 640
BG = (11, 13, 16)
GLOW = (26, 19, 48)
TEXT = (231, 233, 236)
MUTED = (154, 163, 175)
ACCENT = (139, 124, 246)
ACCENT_LIGHT = (201, 194, 255)
ACCENT2 = (94, 203, 176)
BORDER = (38, 43, 51)
PANEL = (22, 26, 32)

FONTS = "C:/Windows/Fonts/"


def font(name, size):
    return ImageFont.truetype(FONTS + name, size)


f_title = font("segoeuib.ttf", 92)
f_tag = font("segoeui.ttf", 30)
f_badge = font("consola.ttf", 20)
f_stat_num = font("consolab.ttf", 64)
f_stat_label = font("segoeui.ttf", 20)
f_url = font("consola.ttf", 26)

img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

# soft radial glow behind the title, approximated with concentric ellipses
glow = Image.new("RGB", (W, H), BG)
gdraw = ImageDraw.Draw(glow)
cx, cy = W // 2, 40
for r, alpha in [(700, 1.0), (560, 0.85), (420, 0.65), (280, 0.4), (160, 0.2)]:
    box = [cx - r, cy - r // 2, cx + r, cy + r]
    shade = tuple(int(BG[i] + (GLOW[i] - BG[i]) * alpha) for i in range(3))
    gdraw.ellipse(box, fill=shade)
glow = glow.filter(ImageFilter.GaussianBlur(60))
img = Image.blend(img, glow, 0.9)
draw = ImageDraw.Draw(img)

# faint node-graph motif on the right, echoing the vault's wikilink graph
import random
random.seed(7)
nodes = []
for _ in range(9):
    nodes.append((random.randint(860, 1190), random.randint(90, 400)))
edges = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (2, 6), (6, 7), (5, 8), (0, 6)]
for a, b in edges:
    draw.line([nodes[a], nodes[b]], fill=BORDER, width=2)
for i, (x, y) in enumerate(nodes):
    r = 6 if i not in (1, 4) else 9
    fill = ACCENT if i in (1, 4) else PANEL
    outline = ACCENT_LIGHT if i in (1, 4) else BORDER
    draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=2)

PAD = 90

# badge pill
badge_text = "LOCAL-FIRST  \u00b7  OPEN SOURCE  \u00b7  MIT LICENSED"
bbox = draw.textbbox((0, 0), badge_text, font=f_badge)
bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
bx, by = PAD, 70
pad_x, pad_y = 22, 14
draw.rounded_rectangle(
    [bx, by, bx + bw + pad_x * 2, by + bh + pad_y * 2],
    radius=22, outline=BORDER, width=2,
)
draw.text((bx + pad_x, by + pad_y - bbox[1]), badge_text, font=f_badge, fill=MUTED)

# title
title_y = 150
draw.text((PAD, title_y), "Mimir", font=f_title, fill=TEXT)

# tagline
tag_y = title_y + 130
tagline = "Your AI agent's memory is a folder of markdown files."
draw.text((PAD, tag_y), tagline, font=f_tag, fill=MUTED)

# stat card, bottom left
stat_y = 430
label_y = stat_y + 90
draw.text((PAD, stat_y), "51.3%", font=f_stat_num, fill=ACCENT2)
draw.text((PAD + 4, label_y), "PERSONAMEM/32K, OFFICIAL AMB HARNESS",
          font=f_stat_label, fill=MUTED)

# divider
div_x = PAD + 400
draw.line([(div_x, stat_y + 10), (div_x, stat_y + 96)], fill=BORDER, width=2)

f_stat_num2 = font("consolab.ttf", 40)
sbbox = draw.textbbox((0, 0), "302/589", font=f_stat_num2)
s_offset = (f_stat_num.getbbox("51.3%")[3] - sbbox[3])  # baseline-align with the big number
draw.text((div_x + 40, stat_y + s_offset), "302/589", font=f_stat_num2, fill=ACCENT_LIGHT)
draw.text((div_x + 44, label_y), "CORRECT ANSWERS",
          font=f_stat_label, fill=MUTED)

# repo url, bottom right
url = "github.com/hasil7677/mimir"
ubbox = draw.textbbox((0, 0), url, font=f_url)
uw = ubbox[2] - ubbox[0]
draw.text((W - PAD - uw, H - 80), url, font=f_url, fill=ACCENT_LIGHT)

img.save("social-preview.png", "PNG")
print("saved social-preview.png", img.size)
