import os
import sys
import logging
import argparse
import c_two as cc

# Import CRM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crms.treeger import Treeger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treeger CRM Launcher')
    parser.add_argument('--server_address', type=str, required=True, help='Memory address for treeger crm server')
    parser.add_argument('--meta_path', type=str, required=True, help='Treeger meta file path')
    args = parser.parse_args()
    
    # Init CRM
    crm = Treeger(meta_path=args.meta_path)
    
    # Launch CRM server
    logger.info('Starting Treeger CRM ...')
    server = cc.rpc.Server(args.server_address, crm)
    server.start()
    logger.info('Treeger CRM started at %s', args.server_address)
    try:
        if server.wait_for_termination():
            server.stop()

    except KeyboardInterrupt:
        server.stop()