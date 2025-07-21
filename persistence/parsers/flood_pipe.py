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

# ==================== Action应用器函数 ====================

def apply_gate_action(parsed_data: dict, action: dict) -> dict:
    """
    应用门控action到gate数据
    
    Args:
        parsed_data: 解析后的数据字典
        action: 要应用的action
        
    Returns:
        应用action后的数据字典
    """
    if "gate" not in parsed_data:
        print("警告: 没有gate数据可供修改")
        return parsed_data
    
    gate_data = parsed_data["gate"]
    action_data = action.get("data", {})
    
    # 修改门高度
    if "gate_height_changes" in action_data:
        changes = action_data["gate_height_changes"]
        for change in changes:
            gate_index = change.get("gate_index")
            new_height = change.get("new_height")
            
            if gate_index is not None and 0 <= gate_index < len(gate_data.gate_height_list):
                old_height = gate_data.gate_height_list[gate_index]
                gate_data.gate_height_list[gate_index] = new_height
                print(f"门 {gate_index} 高度从 {old_height} 修改为 {new_height}")
            else:
                print(f"警告: 无效的门索引 {gate_index}")
    
    # 修改上下游连接
    if "upstream_downstream_changes" in action_data:
        changes = action_data["upstream_downstream_changes"]
        for change in changes:
            gate_index = change.get("gate_index") 
            new_upstream = change.get("new_upstream")
            new_downstream = change.get("new_downstream")
            
            if gate_index is not None and 0 <= gate_index * 2 + 1 < len(gate_data.ud_stream_list):
                if new_upstream is not None:
                    old_upstream = gate_data.ud_stream_list[gate_index * 2]
                    gate_data.ud_stream_list[gate_index * 2] = new_upstream
                    print(f"门 {gate_index} 上游从 {old_upstream} 修改为 {new_upstream}")
                
                if new_downstream is not None:
                    old_downstream = gate_data.ud_stream_list[gate_index * 2 + 1]
                    gate_data.ud_stream_list[gate_index * 2 + 1] = new_downstream
                    print(f"门 {gate_index} 下游从 {old_downstream} 修改为 {new_downstream}")
    
    parsed_data["gate"] = gate_data
    return parsed_data

def apply_rainfall_action(parsed_data: dict, action: dict) -> dict:
    """
    应用降雨action到rainfall数据
    
    Args:
        parsed_data: 解析后的数据字典
        action: 要应用的action
        
    Returns:
        应用action后的数据字典
    """
    if "rainfall" not in parsed_data:
        print("警告: 没有rainfall数据可供修改")
        return parsed_data
    
    rainfall_data = parsed_data["rainfall"]
    action_data = action.get("data", {})
    
    # 修改降雨值
    if "rainfall_changes" in action_data:
        changes = action_data["rainfall_changes"]
        for change in changes:
            time_range = change.get("time_range")  # [start_time, end_time]
            station = change.get("station")
            multiplier = change.get("multiplier", 1.0)  # 乘数
            addition = change.get("addition", 0.0)      # 加数
            
            if time_range and len(time_range) == 2:
                start_time, end_time = time_range
                
                # 找到对应时间范围和站点的数据进行修改
                for i in range(len(rainfall_data.rainfall_date_list)):
                    current_date = rainfall_data.rainfall_date_list[i]
                    current_station = rainfall_data.rainfall_station_list[i]
                    
                    if (station is None or current_station == station) and \
                       start_time <= current_date <= end_time:
                        old_value = rainfall_data.rainfall_value_list[i]
                        new_value = old_value * multiplier + addition
                        rainfall_data.rainfall_value_list[i] = new_value
                        print(f"降雨数据修改: {current_date} {current_station} {old_value} -> {new_value}")
    
    parsed_data["rainfall"] = rainfall_data
    return parsed_data

def apply_tide_action(parsed_data: dict, action: dict) -> dict:
    """
    应用潮位action到tide数据
    
    Args:
        parsed_data: 解析后的数据字典  
        action: 要应用的action
        
    Returns:
        应用action后的数据字典
    """
    if "tide" not in parsed_data:
        print("警告: 没有tide数据可供修改")
        return parsed_data
    
    tide_data = parsed_data["tide"]
    action_data = action.get("data", {})
    
    # 修改潮位值
    if "tide_changes" in action_data:
        changes = action_data["tide_changes"]
        for change in changes:
            time_range = change.get("time_range")  # [start_time, end_time]
            offset = change.get("offset", 0.0)      # 潮位偏移量
            multiplier = change.get("multiplier", 1.0)  # 乘数
            
            if time_range and len(time_range) == 2:
                start_time, end_time = time_range
                
                # 找到对应时间范围的数据进行修改
                for i in range(len(tide_data.tide_date_list)):
                    current_datetime = f"{tide_data.tide_date_list[i]} {tide_data.tide_time_list[i]}"
                    
                    if start_time <= current_datetime <= end_time:
                        old_value = tide_data.tide_value_list[i]
                        new_value = old_value * multiplier + offset
                        tide_data.tide_value_list[i] = new_value
                        print(f"潮位数据修改: {current_datetime} {old_value} -> {new_value}")
    
    parsed_data["tide"] = tide_data
    return parsed_data

# ==================== 数据格式转换函数 ====================

def convert_data_for_model(parsed_data: dict) -> dict:
    """
    将解析后的数据转换为模型需要的格式
    
    Args:
        parsed_data: 解析后的数据字典
        
    Returns:
        转换后的模型输入数据
    """
    model_data = {}
    
    # 转换各种数据类型
    for data_type, data_obj in parsed_data.items():
        if data_type == "ne":
            model_data["ne_data"] = data_obj
        elif data_type == "ns":
            model_data["ns_data"] = data_obj
        elif data_type == "rainfall":
            model_data["rainfall_data"] = data_obj
        elif data_type == "tide":
            model_data["tide_data"] = data_obj
        elif data_type == "gate":
            model_data["gate_data"] = data_obj
        else:
            model_data[data_type] = data_obj
    
    return model_data
