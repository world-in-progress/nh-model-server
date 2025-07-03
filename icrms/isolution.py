import c_two as cc
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union
from pydantic import BaseModel

@dataclass
class NeData:
    grid_id_list: list[int]
    nsl1_list:list[int]
    nsl2_list:list[int]
    nsl3_list:list[int]
    nsl4_list:list[int]
    isl1_list:list[list[int]]
    isl2_list:list[list[int]]
    isl3_list:list[list[int]]
    isl4_list:list[list[int]]
    xe_list:list[float]
    ye_list:list[float]
    ze_list:list[float]
    under_suf_list:list[int]

@dataclass
class NsData:
    edge_id_list: list[int]
    ise_list: list[int]
    dis_list: list[float]
    x_side_list: list[float]
    y_side_list: list[float]
    z_side_list: list[float]
    under_suf_list: list[int]
    nbd_ie_list: list[int]
    ibd_ie_list: list[int]

@dataclass
class RainfallData:
    rainfall_date_list:list[str]
    rainfall_station_list:list[str]
    rainfall_value_list:list[float]

@dataclass
class TideData:
    tide_date_list:list[str]          
    tide_time_list:list[str]           
    tide_value_list:list[float]
    
@dataclass
class Gate:
    ud_stream_list:list[int]
    gate_height_list:list[int]
    grid_id_list: list[list[int]]

@dataclass
class SolutionData:
    ne: NeData
    ns: NsData
    inp: str
    rainfall: RainfallData
    gate: Gate
    tide: TideData

class CreateSolutionBody(BaseModel):
    name: str
    ne_path: str
    ns_path: str
    inp_path: str
    rainfall_path: str
    gate_path: str
    tide_path: str

@cc.icrm
class ISolution:

    # Model Server to Resource Server
    def get_inp(self)-> str:
        """
        获取模型参数
        :return: str
        """
        ...
        
    
    def get_ne(self)-> NeData:
        """
        获取网格数据
        :return: NeData对象列表
        """
        ...
    
    def get_ns(self)-> NsData:
        """
        获取边数据
        :return: NsData对象
        """
        ...
    
    def get_rainfall(self)-> RainfallData:
        """
        获取降雨数据
        :return: RainfallData对象
        """
        ...
    
    def get_gate(self)-> Gate:
        """
        获取闸门数据
        :return:Gate对象
        """
        ...
    
    def get_tide(self)-> TideData:
        """
        获取潮位数据
        :return: TideData对象
        """
        ...

    def get_solution_data(self)-> dict:
        """
        获取解决方案数据
        :return: 解决方案数据
        """
        ...