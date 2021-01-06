import os
import sys
import errno
from glob import glob
import shutil

program = "jupyter nbconvert --CodeFoldingPreprocessor.remove_folded_code=True --TagRemovePreprocessor.remove_cell_tags=\"{'remove_cell_html'}\""
target_default = 'html_with_toclenvs'
targets = {'Book.ipynb':'html'}

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def replace_in_file(fn,src,dest):
    f = open(fn,'r')
    lines = ''.join(f.readlines())
    f.close()
    f = None
    
    f = open(fn+'.tmp','w')
    f.write(lines.replace(src,dest))
    f.close()
    
    os.replace(fn+'.tmp',fn)

for fn in glob("*.ipynb"):
    os.system('%s --to %s "%s"'%(program,targets.get(fn,target_default),fn))

for fn in glob("*.html"):
    replace_in_file(fn,".ipynb)",".html)")
    replace_in_file(fn,".ipynb#",".html#")
