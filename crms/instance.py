import c_two as cc
from icrms.iinstance import IInstance, Signal

@cc.iicrm_instance
class Instance(IInstance):

    def send_signal(self, step: int, signal: Signal) -> dict[str, bool | str]:
        if signal == Signal.START:
            return {'success': True, 'message': 'start signal received'}
        elif signal == Signal.STOP:
            return {'success': True, 'message': 'stop signal received'}
        elif signal == Signal.PAUSE:
            return {'success': True, 'message': 'pause signal received'}
        elif signal == Signal.RESTART:
            return {'success': True, 'message': 'restart signal received'}
        elif signal == Signal.ROLLBACK:
            return {'success': True, 'message': 'rollback signal received'}