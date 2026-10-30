"""Rig the anatomy model: skeleton, skin weights (distance-based, side-restricted, Laplacian-smoothed),
muscle segmentation from the color map, and export to a compact binary + json for three.js."""
import json, sys, colorsys
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix, diags

OUT = sys.argv[1]
m = np.load('mesh.npz'); V = m['V'].astype(np.float64); PVI = m['PVI']; UV = m['UV']; UVI = m['UVI']
corner_v = np.where(PVI < 0, ~PVI, PVI)
quads = corner_v.reshape(-1, 4)            # every polygon is a quad
quv = UVI.reshape(-1, 4)
NV = len(V)

# ---------------- skeleton (model is in cm, Y up, facing +Z; x>0 is the character's left) ----------------
J = {}
def j(name, parent, pos): J[name] = dict(name=name, parent=parent, head=np.array(pos, float))
j('root', None, (0, 92, 0.5))
j('spine', 'root', (0, 104, 0.5))
j('chest', 'spine', (0, 122, 0.5))
j('neck', 'chest', (0, 147.5, -1.5))
j('head', 'neck', (0, 157, 0.5))
for s, sg in (('L', 1), ('R', -1)):
    j('clav' + s, 'chest', (3 * sg, 144, -1))
    j('uarm' + s, 'clav' + s, (17.5 * sg, 142, -1))
    j('farm' + s, 'uarm' + s, (31.5 * sg, 119.5, -1.5))
    j('hand' + s, 'farm' + s, (44.5 * sg, 98, 3.5))
    j('thigh' + s, 'root', (9.5 * sg, 89, 1))
    j('shin' + s, 'thigh' + s, (13.3 * sg, 48, -1))
    j('foot' + s, 'shin' + s, (17.2 * sg, 9, -6.5))
# one finger joint per hand at the knuckles (52% of the way from the wrist to the fingertips), so the hand can
# close around a bar or a ring; the thumb stays on the hand bone
HAND_A = {s: np.array((44.5 * sg, 98, 3.5)) for s, sg in (('L', 1), ('R', -1))}
HAND_B = {s: np.array((48.5 * sg, 76, 8.5)) for s, sg in (('L', 1), ('R', -1))}
KNUCKLE = 0.52
for s in 'LR':
    j('fing' + s, 'hand' + s, HAND_A[s] + (HAND_B[s] - HAND_A[s]) * KNUCKLE)
TAIL = {'head': (0, 179.5, 0.5)}
for s, sg in (('L', 1), ('R', -1)):
    TAIL['hand' + s] = (48.5 * sg, 76, 8.5)
    TAIL['fing' + s] = (48.5 * sg, 76, 8.5)
    TAIL['foot' + s] = (20 * sg, 1, 12)
names = list(J)
kids = {n: [c for c in names if J[c]['parent'] == n] for n in names}
def tail(n):
    if n in TAIL: return np.array(TAIL[n], float)
    # spine-like bones end at their main child; clavicle ends at the upper arm
    main = {'root': 'spine', 'spine': 'chest', 'chest': 'neck', 'neck': 'head'}
    c = main.get(n) or (kids[n][0] if kids[n] else None)
    return J[c]['head']
seg = {n: (J[n]['head'], tail(n)) for n in names}

# ---------------- skin weights ----------------
def seg_dist(P, a, b):
    ab = b - a; t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1), t
D = np.zeros((NV, len(names))); T = np.zeros_like(D)
for i, n in enumerate(names):
    D[:, i], T[:, i] = seg_dist(V, *seg[n])
side = np.sign(V[:, 0])
allowed = np.ones_like(D, bool)
for i, n in enumerate(names):
    if n[-1] in 'LR' and n not in ('root',):
        sg = 1 if n[-1] == 'L' else -1
        allowed[:, i] = side * sg > -0.5           # a limb only takes its own side (inner thighs, arms vs torso)
# the root/pelvis must not grab the arms; arms must not grab the torso below the armpit
armb = [names.index(x + s) for x in ('uarm', 'farm', 'hand') for s in 'LR']
torso_core = np.abs(V[:, 0]) < 19.5
for i in armb: allowed[torso_core & (V[:, 1] < 126), i] = False
legb = [names.index(x + s) for x in ('thigh', 'shin', 'foot') for s in 'LR']
for i in legb: allowed[V[:, 1] > 100, i] = False
for i, n in enumerate(names):
    if n in ('root', 'spine', 'chest', 'neck', 'head', 'clavL', 'clavR'):
        allowed[(np.abs(V[:, 0]) > 28) | (V[:, 1] < 60), i] = False
fingb = [names.index('fing' + s) for s in 'LR']
for i in fingb: allowed[:, i] = False
Dm = np.where(allowed, D, 1e6)
W = 1.0 / np.maximum(Dm, 0.5) ** 4
W[~allowed] = 0
bad = W.sum(1) == 0
print('vertices with no allowed bone:', int(bad.sum()), V[bad][:5] if bad.any() else '')
W[bad] = 1.0 / np.maximum(D[bad], 0.5) ** 4
allowed[bad] = True
W /= W.sum(1, keepdims=True)

# Laplacian smoothing of weights over the mesh (edges of the quads)
e = np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]])
e = np.concatenate([e, e[:, ::-1]])
A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(NV, NV)).tocsr(); A.data[:] = 1
deg = np.asarray(A.sum(1)).ravel(); Lr = diags(1 / np.maximum(deg, 1)) @ A
# keep the middle of long bones firm, smooth more around joints
firm = np.zeros(NV)
for i, n in enumerate(names):
    mid = 1 - np.abs(T[:, i] - 0.5) * 2
    firm = np.maximum(firm, W[:, i] * np.clip(mid, 0, 1))
alpha = 0.6 * (1 - 0.7 * firm)
for it in range(14):
    W = (1 - alpha[:, None]) * W + alpha[:, None] * (Lr @ W)
    W[~allowed] = 0; W /= W.sum(1, keepdims=True)
# fingers: the part of the hand bone's weight beyond the knuckles moves to the finger bone (thumb excluded:
# it sits on the thumb side of the palm, ahead of the other fingers' line)
for s in 'LR':
    hi, fi = names.index('hand' + s), names.index('fing' + s)
    a, b = HAND_A[s], HAND_B[s]; L = np.linalg.norm(b - a); ax = (b - a) / L
    sideax = np.cross(ax, np.array((-0.955 if s == 'L' else 0.955, -0.21, -0.21)))   # thumb-to-little-finger line
    if s == 'R': sideax = -sideax
    t = ((V - a) @ ax) / L; sd = (V - a) @ sideax
    f = np.clip((t - 0.45) / 0.15, 0, 1); f = f * f * (3 - 2 * f)
    f[sd < -6.0] = 0                                      # thumb
    W[:, fi] = W[:, hi] * f; W[:, hi] *= 1 - f
print('finger vertices', int((W[:, names.index('fingL')] > 0.5).sum()), int((W[:, names.index('fingR')] > 0.5).sum()))
# keep the 4 largest influences
idx = np.argsort(-W, 1)[:, :4]; w4 = np.take_along_axis(W, idx, 1)
w4 /= w4.sum(1, keepdims=True)

# ---------------- render vertices: unique (position, uv) pairs ----------------
tris_c = np.concatenate([np.stack([np.arange(len(quads)) * 4 + k for k in (0, 1, 2)], 1),
                         np.stack([np.arange(len(quads)) * 4 + k for k in (0, 2, 3)], 1)])
pv = corner_v; pu = UVI
key = pv.astype(np.int64) * 1_000_000 + pu
uk, inv = np.unique(key, return_inverse=True)
rv = (uk // 1_000_000).astype(np.int64); ru = (uk % 1_000_000).astype(np.int64)
I = inv[tris_c]                                  # triangles in render-vertex indices
P = V[rv]; UVr = UV[ru]
# smooth normals per position
tri_p = pv[tris_c]
fn = np.cross(V[tri_p[:, 1]] - V[tri_p[:, 0]], V[tri_p[:, 2]] - V[tri_p[:, 0]])
N = np.zeros_like(V)
for k in range(3): np.add.at(N, tri_p[:, k], fn)
N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
Nr = N[rv]
Jr = idx[rv]; Wr = w4[rv]
print('render verts', len(P), 'tris', len(I))

# ---------------- muscle segmentation (1024 uv map) ----------------
S = 1024
tex = np.asarray(Image.open('Anatomy_Male_Texture_v001.png').convert('RGB').resize((S, S), Image.NEAREST)).astype(np.float32) / 255
mx = tex.max(2); mn = tex.min(2); sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
r, g, b = tex[..., 0], tex[..., 1], tex[..., 2]
hue = np.zeros_like(mx)
d_ = np.maximum(mx - mn, 1e-6)
hue = np.where(mx == r, ((g - b) / d_) % 6, np.where(mx == g, (b - r) / d_ + 2, (r - g) / d_ + 4)) * 60
beige = (hue > 35) & (hue < 60) & (sat < 0.35)
muscle = (sat > 0.17) & (mx > 0.5) & ~beige
hbin = np.round(hue / 3).astype(int)              # every muscle color is a distinct hue (a few degrees apart)
sbin = (sat > 0.32).astype(int)
cls = np.where(muscle, hbin * 2 + sbin + 1, 0)
lab = np.zeros((S, S), np.int32); nlab = 0
for c in np.unique(cls[cls > 0]):
    l, n = ndimage.label(cls == c)
    if n == 0: continue
    sizes = ndimage.sum(np.ones_like(l), l, np.arange(1, n + 1))
    keep = np.where(sizes >= 60)[0] + 1
    for k in keep:
        nlab += 1; lab[l == k] = nlab
# fill white labels / antialiased edges inside muscles from the nearest muscle (only a few pixels away)
holes = (lab == 0) & (mx > 0.3) & ~beige & ~((sat < 0.05) & (mx > 0.75) & (mx < 0.85))
dist, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
fill = holes & (dist <= 5)
lab[fill] = lab[iy[fill], ix[fill]]
print('muscle components', nlab)

# ---------------- name the muscles by where they sit on the body ----------------
tc = UVr[I].mean(1)                               # triangle uv centroid
px = np.clip((tc[:, 0] * S).astype(int), 0, S - 1); py = np.clip(((1 - tc[:, 1]) * S).astype(int), 0, S - 1)
tl = lab[py, px]
cen = P[I].mean(1); fnr = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]]); area = np.linalg.norm(fnr, axis=1) * .5
# dominant bone per triangle = bone with the biggest summed weight on its 3 vertices
tw = np.zeros((len(I), len(names)))
for k in range(3):
    for q in range(4): np.add.at(tw, (np.arange(len(I)), Jr[I[:, k], q]), Wr[I[:, k], q])
tbone = np.argmax(tw, 1)
GROUPS = ['none', 'chest', 'delts_front', 'delts_side', 'delts_rear', 'traps', 'rhomboids', 'lats', 'serratus', 'abs', 'obliques',
          'erectors', 'biceps', 'triceps', 'forearms', 'glutes', 'quads', 'hamstrings', 'adductors', 'calves', 'shins', 'neck', 'hipflex']
G = {n: i for i, n in enumerate(GROUPS)}
comp_group = np.zeros(nlab + 1, np.uint8)
info = []
for c in range(1, nlab + 1):
    sel = tl == c
    if not sel.any(): continue
    a = area[sel]; ce = (cen[sel] * a[:, None]).sum(0) / a.sum(); nn = (fnr[sel]).sum(0); nn /= np.linalg.norm(nn) + 1e-9
    bones = np.bincount(tbone[sel], weights=a, minlength=len(names)); bn = names[int(np.argmax(bones))]
    x, y, z = ce; ax = abs(x); nx = nn[0] * np.sign(x if x != 0 else 1); nz = nn[2]
    base = bn.rstrip('LR') if bn not in ('root',) else bn
    g = 'none'
    if base in ('head',): g = 'none'
    elif base == 'neck': g = 'traps' if nz < -0.2 else 'neck'
    elif base == 'clav' and ax < 12: g = 'traps' if nz < 0 else 'chest'
    elif base in ('clav',) or (base in ('chest', 'uarm') and ax > 14 and y > 128 and np.linalg.norm(ce - J['uarm' + ('L' if x > 0 else 'R')]['head']) < 11):
        g = 'delts_front' if nz > 0.35 else ('delts_rear' if nz < -0.35 else 'delts_side')
    elif base == 'chest':
        if nz > 0.3: g = 'chest' if y > 118 and ax < 19 else ('serratus' if ax > 10 else 'abs')
        elif nz < -0.3:
            if y > 136 or (ax < 6 and y > 126): g = 'traps'
            elif ax < 9: g = 'rhomboids'
            else: g = 'lats'
        else: g = 'serratus' if nz > -0.05 else 'lats'
    elif base == 'spine':
        if nz > 0.25: g = 'abs' if ax < 7.5 else 'obliques'
        elif nz < -0.25: g = 'erectors' if ax < 8 else 'lats'
        else: g = 'obliques'
    elif base == 'root':
        if nz < -0.15 or (y < 92 and nx > 0.6 and nz < 0.2): g = 'glutes'
        elif nz > 0.3 and ax < 8: g = 'abs'
        else: g = 'hipflex' if y < 95 else 'obliques'
    elif base == 'uarm': g = 'biceps' if nz > 0.15 else ('triceps' if nz < -0.15 else ('biceps' if y > 128 else 'triceps'))
    elif base in ('farm', 'hand'): g = 'forearms' if base == 'farm' else 'none'
    elif base == 'thigh':
        if nz > 0.25: g = 'quads'
        elif nz < -0.35: g = 'glutes' if y > 80 else 'hamstrings'
        elif nx < -0.3: g = 'adductors'
        else: g = 'quads'
    elif base == 'shin': g = 'calves' if nz < 0.1 else 'shins'
    comp_group[c] = G[g]
    info.append((c, g, bn, [round(float(v), 1) for v in ce], [round(float(v), 2) for v in nn], round(float(a.sum()), 1)))
gmap = comp_group[lab]
Image.fromarray(gmap, 'L').save(OUT + '/anatomy_groups.png', optimize=True)
json.dump(info, open('components.json', 'w'), ensure_ascii=False)
from collections import Counter
print(Counter(i[1] for i in info))

# ---------------- export ----------------
bmin = P.min(0); bmax = P.max(0); scl = (bmax - bmin).max()
Pq = np.round((P - bmin) / scl * 65535).astype(np.uint16)
Nq = np.round(Nr * 127).astype(np.int8)
UVq = np.round(np.clip(UVr, 0, 1) * 65535).astype(np.uint16)
Wq = np.round(Wr * 255).astype(np.int32); Wq[:, 0] += 255 - Wq.sum(1); Wq = np.clip(Wq, 0, 255).astype(np.uint8)
blobs = [Pq.tobytes(), Nq.tobytes(), UVq.tobytes(), Jr.astype(np.uint8).tobytes(), Wq.tobytes(), I.astype(np.uint32).tobytes()]
off = []; o = 0
for bl in blobs:
    pad = (-o) % 4; o += pad; off.append(o); o += len(bl)
with open(OUT + '/anatomy.bin', 'wb') as f:
    o = 0
    for bl, oo in zip(blobs, off):
        f.write(b'\0' * (oo - o)); f.write(bl); o = oo + len(bl)
meta = dict(nv=int(len(P)), ni=int(I.size), off=off, bmin=bmin.round(4).tolist(), scale=float(scl),
            bones=[dict(name=n, parent=J[n]['parent'], head=J[n]['head'].round(3).tolist(), tail=tail(n).round(3).tolist()) for n in names],
            groups=GROUPS)
json.dump(meta, open(OUT + '/anatomy.json', 'w'))
# the color map carries white number labels on the muscles: paint them over with the surrounding muscle color
C = np.asarray(Image.open('Anatomy_Male_Texture_v001.png').convert('RGB').resize((2048, 2048), Image.NEAREST)).astype(np.int16)
cmx = C.max(2); cmn = C.min(2)
txt = (cmx > 228) & (cmx - cmn < 40)
txt = ndimage.binary_dilation(txt, iterations=2) & (cmx > 120)
_, (ty, tx) = ndimage.distance_transform_edt(txt, return_indices=True)
C[txt] = C[ty[txt], tx[txt]]
Image.fromarray(C.astype(np.uint8)).save(OUT + '/anatomy_color.jpg', quality=88)
Image.open('Anatomy_Male_Normals_v001.png').convert('RGB').resize((1024, 1024), Image.LANCZOS).save(OUT + '/anatomy_normal.jpg', quality=90)
print('done')
