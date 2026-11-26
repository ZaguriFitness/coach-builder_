import os, sys
from PIL import Image, ImageFilter
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
S = 220
tot = 0
for f in sorted(os.listdir(src)):
    if not f.endswith('.png'): continue
    im = Image.open(os.path.join(src, f)).convert('RGBA')
    m = im.getchannel('A').point(lambda a: 255 if a > 8 else 0)
    # thin things (cables, bands, poles) are ignored when framing, so the body fills the picture
    bb = m.filter(ImageFilter.MinFilter(9)).getbbox() or m.getbbox()
    if bb: bb = (max(0, bb[0]-6), max(0, bb[1]-6), min(im.width, bb[2]+6), min(im.height, bb[3]+6))
    if bb: im = im.crop(bb)
    w, h = im.size; side = int(max(w, h) * 1.08)
    sq = Image.new('RGBA', (side, side), (0, 0, 0, 0))
    sq.paste(im, ((side - w) // 2, (side - h) // 2))
    sq = sq.resize((S, S), Image.LANCZOS)
    out = os.path.join(dst, f[:-4] + '.webp')
    sq.save(out, 'WEBP', quality=72, method=6)
    tot += os.path.getsize(out)
print('files', len(os.listdir(dst)), 'total KB', tot // 1024)
