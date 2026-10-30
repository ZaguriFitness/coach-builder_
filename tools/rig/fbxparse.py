import struct, zlib, numpy as np, sys
def read_fbx(path):
    d=open(path,'rb').read()
    ver=struct.unpack('<I',d[23:27])[0]
    big=ver>=7500
    pos=27
    def prop(p):
        t=chr(d[p]);p+=1
        if t=='Y': return struct.unpack('<h',d[p:p+2])[0],p+2
        if t=='C': return bool(d[p]),p+1
        if t=='I': return struct.unpack('<i',d[p:p+4])[0],p+4
        if t=='F': return struct.unpack('<f',d[p:p+4])[0],p+4
        if t=='D': return struct.unpack('<d',d[p:p+8])[0],p+8
        if t=='L': return struct.unpack('<q',d[p:p+8])[0],p+8
        if t in 'SR':
            n=struct.unpack('<I',d[p:p+4])[0];p+=4;v=d[p:p+n];return (v.decode('utf8','replace') if t=='S' else v),p+n
        if t in 'fdlib':
            n,enc,cl=struct.unpack('<III',d[p:p+12]);p+=12;raw=d[p:p+cl];p+=cl
            if enc==1: raw=zlib.decompress(raw)
            dt={'f':'<f4','d':'<f8','l':'<i8','i':'<i4','b':'?'}[t]
            return np.frombuffer(raw,dtype=dt).copy(),p
        raise ValueError(t)
    def node(p):
        if big: end,np_,pl=struct.unpack('<QQQ',d[p:p+24]);p+=24
        else: end,np_,pl=struct.unpack('<III',d[p:p+12]);p+=12
        nl=d[p];p+=1;name=d[p:p+nl].decode();p+=nl
        if end==0: return None,p
        props=[]
        for _ in range(np_):
            v,p=prop(p);props.append(v)
        kids=[]
        sentinel=25 if big else 13
        while p<end-sentinel or (p<end and end-p>sentinel):
            if p>=end: break
            c,p2=node(p)
            if c is None: p=p2;break
            kids.append(c);p=p2
        return {'name':name,'props':props,'kids':kids},end
    top=[]
    while pos<len(d)-200:
        n,pos=node(pos)
        if n is None: break
        top.append(n)
    return top
def find(n,name): return [k for k in n['kids'] if k['name']==name]
if __name__=='__main__':
    top=read_fbx(sys.argv[1])
    for t in top: print(t['name'],len(t['kids']))
    objs=[t for t in top if t['name']=='Objects'][0]
    from collections import Counter
    print(Counter(k['name'] for k in objs['kids']))
    for k in objs['kids']:
        if k['name'] in('Geometry','Model','Material','Texture','Video'):
            print(k['name'],k['props'][:3] if k['name']!='Video' else k['props'][:3], [c['name'] for c in k['kids']][:25])
