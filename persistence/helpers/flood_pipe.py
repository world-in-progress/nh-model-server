from dataclasses import dataclass
from icrms.isimulation import FenceParams, GateParams, TransferWaterParams

import logging
logger = logging.getLogger(__name__)

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
                row_data = row_data.strip().split(',')
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
                under_suf_list.append(int(float(row_data[-1])))       
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

def is_point_in_polygon(x: float, y: float, polygon_coords: list) -> bool:
    """
    使用射线法判断点是否在多边形内部
    
    Args:
        x: 点的x坐标
        y: 点的y坐标
        polygon_coords: 多边形坐标列表，格式为 [[x1, y1], [x2, y2], ...]
        
    Returns:
        bool: True表示点在多边形内部，False表示在外部
    """
    if len(polygon_coords) < 3:
        return False
    
    n = len(polygon_coords)
    inside = False
    
    p1x, p1y = polygon_coords[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_coords[i % n]
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def is_point_intersects_with_feature(x: float, y: float, feature_json: dict) -> bool:
    """
    判断点是否与GeoJSON feature或FeatureCollection相交
    
    Args:
        x: 点的x坐标
        y: 点的y坐标
        feature_json: GeoJSON格式的地理要素（Feature或FeatureCollection）
        
    Returns:
        bool: True表示相交，False表示不相交
    """
    if not feature_json:
        return False
    
    # 检查是否是FeatureCollection
    if feature_json.get('type') == 'FeatureCollection':
        features = feature_json.get('features', [])
        # 只要与任何一个feature相交就返回True
        for feature in features:
            if is_point_intersects_with_feature(x, y, feature):
                return True
        return False
    
    # 处理单个Feature
    if 'geometry' not in feature_json:
        return False
    
    geometry = feature_json['geometry']
    geom_type = geometry.get('type', '').lower()
    coordinates = geometry.get('coordinates', [])
    
    if geom_type == 'polygon':
        # 对于Polygon，coordinates是 [外环, 内环1, 内环2, ...]
        if not coordinates:
            return False
        
        # 检查是否在外环内
        exterior_ring = coordinates[0]
        if not is_point_in_polygon(x, y, exterior_ring):
            return False
        
        # 检查是否在任何内环（洞）内，如果在洞内则不相交
        for i in range(1, len(coordinates)):
            interior_ring = coordinates[i]
            if is_point_in_polygon(x, y, interior_ring):
                return False
        
        return True
    
    elif geom_type == 'multipolygon':
        # 对于MultiPolygon，coordinates是 [polygon1, polygon2, ...]
        for polygon_coords in coordinates:
            if not polygon_coords:
                continue
            
            # 检查是否在外环内
            exterior_ring = polygon_coords[0]
            if not is_point_in_polygon(x, y, exterior_ring):
                continue
            
            # 检查是否在任何内环（洞）内
            in_hole = False
            for i in range(1, len(polygon_coords)):
                interior_ring = polygon_coords[i]
                if is_point_in_polygon(x, y, interior_ring):
                    in_hole = True
                    break
            
            if not in_hole:
                return True
        
        return False
    
    elif geom_type == 'point':
        # 对于Point，检查是否是同一个点（考虑浮点数精度）
        if len(coordinates) >= 2:
            return abs(coordinates[0] - x) < 1e-9 and abs(coordinates[1] - y) < 1e-9
        return False
    
    elif geom_type == 'linestring':
        # 对于LineString，检查点是否在线上（这里使用简单的距离判断）
        # 实际应用中可能需要更复杂的线段相交算法
        if len(coordinates) < 2:
            return False
        
        tolerance = 1e-6  # 容差
        for i in range(len(coordinates) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            
            # 使用点到线段的距离判断
            # 这里简化处理，实际可能需要更精确的算法
            if abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / \
               ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5 < tolerance:
                return True
        
        return False
    
    # 其他几何类型暂不支持
    return False

def apply_add_fence_action(fence_params: FenceParams, model_data: dict) -> dict:

    print("----------------------1234567890----------------------")
    elevation_delta = fence_params.elevation_delta
    landuse_type = fence_params.landuse_type
    feature_json = fence_params.feature

    print(f"elevation_delta: {elevation_delta}")
    print(f"landuse_type: {landuse_type}")
    print(f"feature_json: {feature_json}")

    ne_data: NeData = model_data.get('ne', {})
    ns_data: NsData = model_data.get('ns', {})
    
    print("开始遍历ne_data")
    for index in range(len(ne_data.xe_list)):
        x = ne_data.xe_list[index]
        y = ne_data.ye_list[index]
        
        # 判断当前网格点是否与feature相交
        if is_point_intersects_with_feature(x, y, feature_json):
            if elevation_delta is not None:
                ne_data.ze_list[index] += elevation_delta
            if landuse_type is not None:
                ne_data.under_suf_list[index] = landuse_type
            logger.info(f"网格中心点 ({x}, {y}) 与feature相交，应用了地形变化: {elevation_delta} 和土地利用类型: {landuse_type}")

    print("开始遍历ns_data")
    for index in range(len(ns_data.x_side_list)):
        x = ns_data.x_side_list[index]
        y = ns_data.y_side_list[index]

        # 判断当前网格点是否与feature相交
        if is_point_intersects_with_feature(x, y, feature_json):
            if elevation_delta is not None:
                ns_data.z_side_list[index] += elevation_delta
            if landuse_type is not None:
                ns_data.s_type_list[index] = landuse_type
            logger.info(f"边中心点 ({x}, {y}) 与feature相交，应用了地形变化: {elevation_delta} 和土地利用类型: {landuse_type}")

    model_data['ne'] = ne_data
    model_data['ns'] = ns_data
    
    return model_data

def apply_add_gate_action(gate_params: GateParams, model_data: dict) -> dict:
    """应用闸门动作"""
    print(f"应用闸门动作: {gate_params}")
    return model_data

def apply_transfer_water_action(transfer_water_params: TransferWaterParams, model_data: dict) -> dict:
    """应用调水动作"""
    print(f"应用调水动作: {transfer_water_params}")
    return model_data
