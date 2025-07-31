import os
import sys
import json
import logging
import argparse
import c_two as cc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crms.simulation import Simulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulation Launcher')
    parser.add_argument('--timeout', type=int, help='Timeout for the server to start (in seconds)')
    parser.add_argument('--server_address', type=str, required=True, help='TCP address for the server')
    parser.add_argument('--solution_node_key', type=str, required=True, help='Solution name')
    parser.add_argument('--simulation_node_key', type=str, required=True, help='Simulation name')
    parser.add_argument('--process_group_config', type=json.loads, required=True, help='Process group configuration')

    args = parser.parse_args()
    
    server_address = args.server_address

    crm = Simulation(args.solution_node_key, args.simulation_node_key, args.process_group_config)
    server = cc.rpc.Server(server_address, crm)
    server.start()
    logger.info(f'Starting CRM server at {server_address}')
    try:
        if server.wait_for_termination(None if (args.timeout == -1 or args.timeout == 0) else args.timeout):
            logger.info('Timeout reached, terminating Simulation CRM...')
            server.stop()
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt received, terminating CRM...')
        server.stop()
    finally:
        logger.info('CRM terminated.')

