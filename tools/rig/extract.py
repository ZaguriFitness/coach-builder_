import sys,numpy as np
sys.path.insert(0,'.')
from fbxparse import read_fbx,find
top=read_fbx(sys.argv[1]);objs=[t for t in top if t['name']=='Objects'][0]
for v in [k for k in objs['kids'] if k['name']=='Video']:
    c=find(v,'Content')
    name=v['props'][1].split('\x00')[0]
    if c and isinstance(c[0]['props'][0],(bytes,bytearray)) and len(c[0]['props'][0])>0:
        open(name,'wb').write(c[0]['props'][0]);print('saved',name,len(c[0]['props'][0]))
    else: print('no content',name)
g=[k for k in objs['kids'] if k['name']=='Geometry'][0]
V=find(g,'Vertices')[0]['props'][0].reshape(-1,3)
PVI=find(g,'PolygonVertexIndex')[0]['props'][0]
uvl=find(g,'LayerElementUV')[0]
for c in uvl['kids']: print(' uv',c['name'],c['props'][:1] if c['name'] not in('UV','UVIndex') else len(c['props'][0]))
UV=find(uvl,'UV')[0]['props'][0].reshape(-1,2); UVI=find(uvl,'UVIndex')[0]['props'][0]
nl=find(g,'LayerElementNormal')[0]
for c in nl['kids']: print(' n',c['name'],c['props'][:1] if c['name'] not in('Normals',) else len(c['props'][0]))
mdl=[k for k in objs['kids'] if k['name']=='Model'][0]
for p in find(find(mdl,'Properties70')[0],'P'): print(' model P',p['props'][0],p['props'][4:])
gs=[t for t in top if t['name']=='GlobalSettings'][0]
for p in find(find(gs,'Properties70')[0],'P'): print(' gs',p['props'][0],p['props'][4:])
print('verts',V.shape,'pvi',PVI.shape,'uv',UV.shape,'bbox',V.min(0),V.max(0))
# polygons
polys=[];cur=[]
for i,x in enumerate(PVI):
    if x<0: cur.append(~x);polys.append(cur);cur=[]
    else: cur.append(x)
from collections import Counter
print('polys',len(polys),Counter(len(p) for p in polys).most_common(5))
np.savez('mesh.npz',V=V,PVI=PVI,UV=UV,UVI=UVI)
