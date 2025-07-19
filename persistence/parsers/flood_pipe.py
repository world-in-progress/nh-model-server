from pathlib import Path
from dataclasses import dataclass
import os

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
    ise_list: list[list[int]]
    dis_list: list[float]
    x_side_list: list[float]
    y_side_list: list[float]
    z_side_list: list[float]
    s_type_list: list[int]

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

def get_ne(ne_path) -> NeData:
        grid_id_list = [0]
        nsl1_list = [0]
        nsl2_list = [0]
        nsl3_list = [0]
        nsl4_list = [0]
        isl1_list = [[0,0,0,0,0,0,0,0,0,0]]
        isl2_list = [[0,0,0,0,0,0,0,0,0,0]]
        isl3_list = [[0,0,0,0,0,0,0,0,0,0]]
        isl4_list = [[0,0,0,0,0,0,0,0,0,0]]
        xe_list = [0.0]
        ye_list = [0.0]
        ze_list = [0.0]
        under_suf_list = [0]
        with open(ne_path, 'r', encoding='utf-8') as f:
            for row_data in f:
                row_data = row_data.split(',')
                # 创建NeData对象
                grid_id_list.append(int(row_data[0]))
                nsl1 = int(row_data[1])
                nsl2 = int(row_data[2])
                nsl3 = int(row_data[3])
                nsl4 = int(row_data[4])
                nsl1_list.append(nsl1)
                nsl2_list.append(nsl2)
                nsl3_list.append(nsl3)
                nsl4_list.append(nsl4)
                isl1 = [0 for _ in range(nsl1)]
                isl2 = [0 for _ in range(nsl2)]
                isl3 = [0 for _ in range(nsl3)]
                isl4 = [0 for _ in range(nsl4)]
                for i in range(nsl1): 
                    isl1[i] = int(row_data[5+i]) 
                for i in range(nsl2): 
                    isl2[i] = int(row_data[5+nsl1+i])
                for i in range(nsl3): 
                    isl3[i] = int(row_data[5+nsl1+nsl2+i])
                for i in range(nsl4): 
                    isl4[i] = int(row_data[5+nsl1+nsl2+nsl3+i])
                isl1_list.append(isl1)
                isl2_list.append(isl2)
                isl3_list.append(isl3)
                isl4_list.append(isl4)
                xe_list.append(float(row_data[-4]))
                ye_list.append(float(row_data[-3]))
                ze_list.append(float(row_data[-2]))
                under_suf_list.append(int(row_data[-1]))       
        ne_data = NeData(grid_id_list,nsl1_list,nsl2_list,nsl3_list,nsl4_list,isl1_list,isl2_list,isl3_list,isl4_list,xe_list,ye_list,ze_list,under_suf_list)
        return ne_data
    
def get_ns(ns_path) -> NsData:
    edge_id_list = [0]
    ise_list = [[0,0,0,0,0]]
    dis_list = [0.0]
    x_side_list = [0.0]
    y_side_list = [0.0]
    z_side_list = [0.0]
    s_type_list = [0]
    with open(ns_path,'r',encoding='utf-8') as f:
        for rowdata in f:
            ise_row = []
            rowdata = rowdata.strip().split(",")
            edge_id_list.append(int(float(rowdata[0].strip())))
            ise_row = [
                int(rowdata[1].strip()),
                int(rowdata[2].strip()),
                int(rowdata[3].strip()),
                int(rowdata[4].strip()),
                int(rowdata[5].strip())
            ]
            ise_list.append(ise_row)
            dis_list.append(float(rowdata[6].strip()))
            x_side_list.append(float(rowdata[7].strip()))
            y_side_list.append(float(rowdata[8].strip()))
            z_side_list.append(float(rowdata[9].strip()))
            s_type_list.append(float(rowdata[10].strip()))
    ns_data = NsData(
        edge_id_list,
        ise_list,
        dis_list,
        x_side_list,
        y_side_list,
        z_side_list,
        s_type_list
    )
    return ns_data

def get_rainfall(rainfall_path) -> RainfallData:
    rainfall_date_list = []
    rainfall_station_list = []
    rainfall_value_list = []
    with open(rainfall_path,'r',encoding='utf-8') as f:
        # 跳过第一行
        next(f)
        for row_data in f:
            row_data = row_data.split(',')
            rainfall_date_list.append(row_data[0])
            rainfall_station_list.append(row_data[1])
            rainfall_value_list.append(float(row_data[2]))
    rainfall = RainfallData(
        rainfall_date_list,
        rainfall_station_list,
        rainfall_value_list
    )
    return rainfall

def get_gate(gate_path) -> Gate:
    ud_stream_list = []
    gate_height_list = []
    grid_id_list = []
    with open(gate_path,'r',encoding='utf-8') as f:
        for row_data in f:
            row_data = row_data.strip().split(',')
            ud_stream_list.append(int(row_data[0]))
            ud_stream_list.append(int(row_data[1]))
            gate_height_list.append(int(row_data[2]))
            grid_id_row = []
            for value in row_data[3:]:
                grid_id_row.append(int(value))
            grid_id_list.append(grid_id_row)
    gate = Gate(
        ud_stream_list=ud_stream_list,
        gate_height_list=gate_height_list,
        grid_id_list=grid_id_list
    )
    return gate

def get_tide(tide_path) -> TideData:
    tide_date_list = []
    tide_time_list = []
    tide_value_list = []
    with open(tide_path,'r',encoding='utf-8') as f:
        # 跳过第一行
        next(f)
        for row_data in f:
            row_data = row_data.split(',')
            tide_date_list.append(row_data[0])
            tide_time_list.append(row_data[1])
            tide_value_list.append(float(row_data[2]))
    tide = TideData(
        tide_date_list,
        tide_time_list,
        tide_value_list
    )
    return tide
