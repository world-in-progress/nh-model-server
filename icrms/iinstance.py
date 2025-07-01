import c_two as cc
from enum import Enum
from pydantic import BaseModel

class Signal(str, Enum):
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESTART = "restart"
    ROLLBACK = "rollback"

@cc.icrm
class IInstance(BaseModel):
    
    def send_signal(self, step: int, signal: Signal) -> dict[str, bool | str]:
        """
        发送信号
        :param signal: 信号
        :return: 信号
        """
        ...