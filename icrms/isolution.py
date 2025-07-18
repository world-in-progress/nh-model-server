import c_two as cc

@cc.icrm
class ISolution:

    # Model Server to Resource Server
    def clone_env(self) -> dict:
        """
        克隆环境变量
        :return: 环境变量
        """
        ...

    def get_env(self) -> dict:
        """
        获取环境变量字典
        :return: 环境变量字典
        """
        ...

    def get_action_types(self) -> list[str]:
        """
        获取动作类型列表
        :return: 动作类型列表
        """
        ...

    def add_human_action(self, action_type: str, params: dict) -> str:
        """
        添加人工操作
        :param action: 人工操作参数
        :return: 添加结果
        """
        ...

    def get_human_actions(self) -> list[dict]:
        """
        获取所有人工操作
        :return: 人工操作列表
        """
        ...