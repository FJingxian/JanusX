#!/user/bin/env python
# -*- coding: utf-8 -*-
import sys
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*ChainedAssignmentError.*"
)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.backends.backend_pdf # pdf support
from . import gwas2,gs,postGWAS,grm,pca,gformat,phylogeny,admixture

__logo__ = r'''
       _                      __   __
      | |                     \ \ / /
      | | __ _ _ __  _   _ ___ \ V / 
  _   | |/ _` | '_ \| | | / __| > <  
 | |__| | (_| | | | | |_| \__ \/ . \ 
  \____/ \__,_|_| |_|\__,_|___/_/ \_\
'''
__version__ = 'JanusX v1.0.0'

def main():
    module = dict(zip(['gwas','gs','postGWAS','grm','pca','gformat','phylogeny','admixture'],
                      [gwas2,gs,postGWAS,grm,pca,gformat,phylogeny,admixture]))
    extmodule = {}
    print(__logo__)
    if len(sys.argv)>1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            print('Usage: jx <module> [parameter]')
            print(f'''Modules: {' '.join(module.keys())}''')
            print(f'''extModules: {' '.join(extmodule.keys())}''')
        elif sys.argv[1] == '-v' or sys.argv[1] == '--version':
            print(__version__)
        else:
            module_name = sys.argv[1]
            module.update(extmodule)
            if sys.argv[1] in module.keys():
                sys.argv.remove(sys.argv[1])
                module[module_name].main() # Process of Target Module
            elif sys.argv[1] not in module.keys():
                print(f'Unkown module: {sys.argv[1]}')
                print(f'Usage: {sys.argv[0]} <module> [parameter]')
                print(f'''Modules: {' '.join(module.keys())}''')
                print(f'''extModules: {' '.join(extmodule.keys())}''')
    else:
        print(f'Usage: {sys.argv[0]} <module> [parameter]')
        print(f'''Modules: {' '.join(module.keys())}''')
        print(f'''extModules: {' '.join(extmodule.keys())}''')

if __name__ == "__main__":
    main()