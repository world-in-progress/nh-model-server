from __future__ import annotations
import os
import time
import yaml
import shutil
import logging
import threading
import c_two as cc
from contextlib import contextmanager
from typing import Generator, Type, TypeVar

from ..core.config import settings
from crms.treeger import ITreeger, Treeger, TreeMeta, ReuseAction, CRMDuration

# Configure logging
logger = logging.getLogger('BSTreeger')

T = TypeVar('T')

class BootStrappingTreeger:
    instance: ITreeger | 'BootStrappingTreeger' = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls.instance is None:
            with cls._lock:
                if cls.instance is None:
                    cls.instance = super(BootStrappingTreeger, cls).__new__(cls)
                    cls.instance._initialized = False
        return cls.instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        
        # Pre-remove memory temp directory if it exists
        if (
            settings.MEMORY_TEMP_DIR 
            and settings.PRE_REMOVE_MEMORY_TEMP_DIR
            and os.path.exists(settings.MEMORY_TEMP_DIR)
            ):
            try:
                shutil.rmtree(settings.MEMORY_TEMP_DIR)
            except OSError as e:
                logger.error(f'Failed to remove memory temp directory: {e}')
        
        self._meta_path = settings.SCENARIO_META_PATH
        self._server_address = settings.TREEGER_SERVER_ADDRESS
        
        self._crm_server = cc.rpc.Server(self._server_address, Treeger(self._meta_path))

        if not self._meta_path or not self._server_address:
            raise ValueError('Treeger meta path and server address must be set in settings')

        try:
            with open(self._meta_path, 'r') as f:
                meta_data = yaml.safe_load(f)
            
            meta = TreeMeta(**(meta_data['meta']))
            for crm_temp in meta.crm_entries:
                if crm_temp.name == 'Treeger':
                    self._crm_launcher = crm_temp.crm_launcher
                    break
            
            # Initialize the CRM process
            self._bootstrap()
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f'Failed to initialize Treeger from {self._meta_path}: {e}')
            self._initialized = False
            raise
        
    def _bootstrap(self):
        self._crm_server.start()
        
        start_time = time.time()
        timeout = 60
        
        logger.info('Waiting for Treeger CRM to start...')
        while True:
            if cc.rpc.Client.ping(self._server_address, timeout=1) is False:
                if time.time() - start_time > timeout:
                    logger.error(f'Timeout waiting for Treeger CRM to start after {timeout} seconds')
                    raise TimeoutError(f'Treeger CRM failed to start within {timeout} seconds')
            else:
                logger.info('Treeger CRM started successfully')
                break
    
    def terminate(self):
        while cc.rpc.Client.shutdown(self._server_address) is False:
            time.sleep(1)
        logger.info('Treeger CRM shutdown successfully')
    
    def __getattr__(self, name):
        icrm = ITreeger()
        if hasattr(icrm, name):
            def method_wrapper(*args, **kwargs):
                with cc.compo.runtime.connect_crm(self._server_address, ITreeger) as crm:
                    remote_method = getattr(crm, name)
                    return remote_method(*args, **kwargs)
            return method_wrapper
        else:
            logger.error(f'Attribute {name} not found in ITreeger')
            raise AttributeError(f'{name} not found in ITreeger')
        
    @contextmanager
    def connect(self, node_key: str, icrm: Type[T], duration: CRMDuration = CRMDuration.Medium) -> Generator[T, None, None]:
        proxy_crm = None
        try:
            with cc.compo.runtime.connect_crm(self._server_address, ITreeger) as crm:
                server_address = crm.activate_node(node_key, ReuseAction.KEEP, duration)
                
            client = cc.rpc.Client(server_address)
            proxy_crm = icrm()
            proxy_crm.client = client
            yield proxy_crm
            
        finally:
            try:
                # Terminate the CRM server process
                if duration == CRMDuration.Once:
                    with cc.compo.runtime.connect_crm(self._server_address, ITreeger) as crm:
                        crm.deactivate_node(node_key)
                    
                # Terminate the client connection
                if proxy_crm and hasattr(proxy_crm, 'client'):
                    proxy_crm.client.terminate()
            except Exception as e:
                logger.warning(f'Failed to disconnect services from node "{node_key}": {e}')

BT = BootStrappingTreeger