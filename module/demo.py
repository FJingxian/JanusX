import argparse
import time
import socket
import logging
import sys
import os
'''
Examples:
  # Basic usage with xxx
  --param1 genotypes.vcf --param2 phenotypes.txt --param3 results

File Formats:
  xxx:    xxx (bed/bim/fam)
        
Citation:
  https://github.com/MaizeMan-JxFU/xxx/
'''
def setup_logging(log_file_path):
    """set logging"""
    if os.path.exists(log_file_path) and log_file_path[-4:]=='.log':
        os.remove(log_file_path)
    # creart logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # clean exist handler
    logger.handlers.clear()
    # set log format
    formatter = logging.Formatter()
    # file handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    # handler of control panel
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    # add handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
def main(log:bool=True):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument('--param1', type=str, 
                           help='xxx')
    geno_group.add_argument('--param2', type=str, 
                           help='xxx')
    required_group.add_argument('--param3', type=str, required=True,
                               help='xxx')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('--out', type=str, default='test',
                               help='xxx'
                                   '(default: %(default)s)')
    args = parser.parse_args()
    # Build argument list for the original script
    sys.argv = [
        sys.argv[0],  # script name
        args.param1,
        args.param2,
        args.param3,
        args.out,
    ]
    # create log file
    folder = os.path.dirname(args.out)
    folder = '.' if folder =='' else folder
    prefix = os.path.basename(args.out)
    if not os.path.exists(folder):
        os.mkdir(folder,0o755)
    
    logger = setup_logging(f'''{args.out}.log'''.replace('//','/'))
    logger.info('High Performance Linear Mixed Model Solver for Genome-Wide Association Studies')
    logger.info(f'Host: {socket.gethostname()}\n')
    # Print configuration summary
    if log:
        logger.info("*"*60)
        logger.info("xxx")
        logger.info("*"*60)
        logger.info(f"param1:         {args.param1}")
        logger.info(f"output path:      {folder}")
        logger.info(f"output prefix:      {prefix}")
        logger.info("*"*60 + "\n")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.out):
        os.makedirs(args.out, mode=0o755)
        if log:
            print(f"Created output directory: {args.out}")
    return args,logger

t_start = time.time()
args,logger = main()

lt = time.localtime()
endinfo = f'\nFinished, Total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
logger.info(endinfo)