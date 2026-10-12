import struct, sys
o = open(r'<repo>/examples/p10/mmap_eq.vmb','rb').read()
w = open(r'<repo>/examples/p10/mmap_eq_w.vmbw','rb').read()
c = open(r'<repo>/examples/p10/mmap_eq_code.vmb','rb').read()
co, cl = struct.unpack_from('<II', o, 16)
orig_co, orig_cl = struct.unpack_from('<II', o, 16)
o_co, o_cl = struct.unpack_from('<II', o, 16)
ok_w = (o[co:co+cl] == w)
mo_orig = struct.unpack_from('<I', o, 24)[0]
mo_new  = struct.unpack_from('<I', c, 24)[0]
ok_text = (c[32:32+struct.unpack_from('<I',o,12)[0]] ==
           o[32:32+struct.unpack_from('<I',o,12)[0]])
ml = struct.unpack_from('<I', o, 28)[0]
ok_meta = (c[mo_new:mo_new+ml] == o[mo_orig:mo_orig+ml])
print('WEIGHTS_EXACT=%d TEXT_EXACT=%d META_EXACT=%d' % (ok_w, ok_text, ok_meta))
