import sys
if sys.version_info[0] < 3:
    raise ImportError("Need Python 3.x")

def obfuscate(s):
    byt = s.encode()
    mask = b'keyword'
    lmask = len(mask)
    return bytes([c ^ mask[i % lmask] for i, c in enumerate(byt)])

def deobfuscate(byt):
    mask = b'keyword'
    lmask = len(mask)
    byt = bytes([c ^ mask[i % lmask] for i, c in enumerate(byt)])
    return byt.decode()

if __name__ == '__main__':
    import os
    if len(sys.argv) <= 1:
        print("USAGE: python obfuscate.py [-d] file")
        exit(0)
    if sys.argv[1] == '-d':
        #deobfuscate
        with open(sys.argv[2],'rb') as f:
            data = f.read()
        s = deobfuscate(data)
        base,ext = os.path.splitext(sys.argv[2])
        fn = base+'_decoded'+ext
        print("Saving to",fn)
        with open(fn,'w') as f:
            f.write(s)
    else:
        #obfuscate
        with open(sys.argv[1],'r') as f:
            data = ''.join(f.readlines())
        byt = obfuscate(data)
        base,ext = os.path.splitext(sys.argv[1])
        fn = base+'_encoded'+ext
        print("Saving to",fn)
        with open(fn,'wb') as f:
            f.write(byt)
