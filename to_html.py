from __future__ import print_function

import os
import sys
import errno
from glob import glob
import shutil
import time

OUTPUT_DIR = 'html'
if len(sys.argv) > 1:
    OUTPUT_DIR = sys.argv[1]
print("OUTPUTTING TO",OUTPUT_DIR)
time.sleep(1)

program = "jupyter nbconvert --CodeFoldingPreprocessor.remove_folded_code=True --TagRemovePreprocessor.remove_cell_tags=\"{'remove_cell_html'}\""
target_default = 'html_toc'
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
    print("Running on ",fn)
    os.system('%s --to %s --output-dir=%s "%s"'%(program,targets.get(fn,target_default),OUTPUT_DIR,fn))

for fn in glob(OUTPUT_DIR+"/*.html"):
    replace_in_file(fn,".ipynb\"",".html\"")
    replace_in_file(fn,".ipynb#",".html#")

eqn_numbering_location = """<!-- End of mathjax configuration --></head>"""

inject_eqn_numbering = """
<script type="text/x-mathjax-config">
// make sure that equations numbers are enabled
MathJax.Hub.Config({ TeX: { equationNumbers: {
    autoNumber: "all", // All AMS equations are numbered
    useLabelIds: true, // labels as ids
    // format the equation number - uses an offset eqNumInitial (default 0)
    formatNumber: function (n) {return String(Number(n)+Number(1)-1)} 
    } } 
});
</script>
"""

for fn in glob(OUTPUT_DIR+"/*.html"):
    replace_in_file(fn,eqn_numbering_location,inject_eqn_numbering+eqn_numbering_location)