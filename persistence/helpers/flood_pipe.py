from dataclasses import dataclass
from icrms.isimulation import FenceParams, GateParams, TransferWaterParams

import math
import logging
import numpy as np
from pyproj import Transformer
logger = logging.getLogger(__name__)

# 全局缓存的坐标转换器
_transformer_4326_to_2326 = None

def get_transformer_4326_to_2326():
    """获取缓存的坐标转换器"""
    global _transformer_4326_to_2326
    if _transformer_4326_to_2326 is None:
        _transformer_4326_to_2326 = Transformer.from_crs("EPSG:4326", "EPSG:2326", always_xy=True)
    return _transformer_4326_to_2326

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

def get_ne(ne_path: str) -> NeData:
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
            isl1 = [0 for _ in range(10)]
            isl2 = [0 for _ in range(10)]
            isl3 = [0 for _ in range(10)]
            isl4 = [0 for _ in range(10)]
            for i in range(nsl1): 
                isl1[i+1] = int(row_data[5+i]) 
            for i in range(nsl2): 
                isl2[i+1] = int(row_data[5+nsl1+i])
            for i in range(nsl3): 
                isl3[i+1] = int(row_data[5+nsl1+nsl2+i])
            for i in range(nsl4): 
                isl4[i+1] = int(row_data[5+nsl1+nsl2+nsl3+i])
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

def get_ns(ns_path: str) -> NsData:
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

def get_rainfall(rainfall_path: str) -> RainfallData:
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

def get_gate(gate_path: str) -> Gate:
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

def get_tide(tide_path: str) -> TideData:
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

def is_point_intersects_with_feature(x: float, y: float, feature_json: dict, ne_data: NeData = None) -> bool:
    """
    判断点是否与GeoJSON feature或FeatureCollection相交
    
    Args:
        x: 点的x坐标
        y: 点的y坐标
        feature_json: GeoJSON格式的地理要素（Feature或FeatureCollection）
        ne_data: 网格数据，用于动态计算缓冲区距离
        
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
            if is_point_intersects_with_feature(x, y, feature, ne_data):
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
        # 对于LineString，动态计算缓冲区距离
        if len(coordinates) < 2:
            return False
        
        # 动态计算缓冲区距离
        buffer_distance = calculate_dynamic_buffer_distance(x, y, ne_data)
        
        for i in range(len(coordinates) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            
            # 计算点到线段的最短距离
            distance = point_to_line_segment_distance(x, y, x1, y1, x2, y2)
            
            # 如果距离小于动态计算的缓冲区距离，认为相交
            if distance <= buffer_distance:
                return True
        
        return False
    
    # 其他几何类型暂不支持
    return False

def calculate_dynamic_buffer_distance(x: float, y: float, ne_data: NeData) -> float:
    """
    动态计算缓冲区距离，基于当前点与最近邻网格点的距离
    
    Args:
        x: 当前点的x坐标
        y: 当前点的y坐标
        ne_data: 网格数据
        
    Returns:
        float: 动态计算的缓冲区距离
    """
    if not ne_data or len(ne_data.xe_list) < 2:
        return 50.0  # 默认值
    
    min_distance = float('inf')
    
    # 找到最近的邻居网格点
    for i in range(len(ne_data.xe_list)):
        grid_x = ne_data.xe_list[i]
        grid_y = ne_data.ye_list[i]
        
        # 跳过当前点本身
        if abs(grid_x - x) < 1e-6 and abs(grid_y - y) < 1e-6:
            continue
            
        distance = math.sqrt((x - grid_x)**2 + (y - grid_y)**2)
        if distance < min_distance:
            min_distance = distance
    
    # 使用最近邻距离的一半作为缓冲区距离
    # 这样可以确保不会过度扩大影响范围
    return min_distance / 2.0 if min_distance != float('inf') else 50.0

def point_to_line_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """
    计算点到线段的最短距离
    
    Args:
        px, py: 点坐标
        x1, y1: 线段起点
        x2, y2: 线段终点
        
    Returns:
        float: 点到线段的最短距离
    """
    # 线段向量
    dx = x2 - x1
    dy = y2 - y1
    
    # 如果线段退化为点
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    
    # 计算点在线段上的投影参数t (0 <= t <= 1表示投影在线段上)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    # 计算投影点坐标
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    # 计算点到投影点的距离
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

def transform_coordinates_4326_to_2326(lon: float, lat: float) -> tuple[float, float]:
    """
    快速坐标转换（使用缓存的转换器）
    
    Args:
        lon: 经度 (EPSG:4326)
        lat: 纬度 (EPSG:4326)
        
    Returns:
        tuple[float, float]: 转换后的坐标 (x, y) in EPSG:2326
    """
    transformer = get_transformer_4326_to_2326()
    x, y = transformer.transform(lon, lat)
    return x, y

def transform_point_list_4326_to_2326(point_list: list) -> list:
    """
    快速坐标点列表转换（使用缓存的转换器）
    
    Args:
        point_list: 坐标点列表 [lon, lat] in EPSG:4326
        
    Returns:
        list: 转换后的坐标点列表 [x, y] in EPSG:2326
    """
    if not isinstance(point_list, list) or len(point_list) < 2:
        return point_list
    
    lon, lat = point_list[0], point_list[1]
    x, y = transform_coordinates_4326_to_2326(lon, lat)
    
    return [x, y]

def transform_feature_4326_to_2326(feature: dict) -> dict:
    """
    将GeoJSON feature从EPSG:4326转换为EPSG:2326
    
    Args:
        feature: GeoJSON格式的地理要素（Feature或FeatureCollection）
        
    Returns:
        dict: 转换后的GeoJSON feature
    """
    if not feature:
        return feature
        
    def transform_coordinates(coords):
        if isinstance(coords[0], (int, float)):
            # 单个坐标点
            x, y = transform_coordinates_4326_to_2326(coords[0], coords[1])
            return [x, y]
        return [transform_coordinates(c) for c in coords]
    
    # 深拷贝以避免修改原始数据
    import copy
    feature = copy.deepcopy(feature)
    
    # 处理FeatureCollection
    if feature.get('type') == 'FeatureCollection':
        for f in feature.get('features', []):
            if 'geometry' in f:
                f['geometry']['coordinates'] = transform_coordinates(f['geometry']['coordinates'])
        return feature
    
    # 处理单个Feature
    if 'geometry' in feature:
        feature['geometry']['coordinates'] = transform_coordinates(feature['geometry']['coordinates'])
    
    return feature

def find_grid_for_point(x: float, y: float, ne_data: NeData) -> int | None:
    """
    根据坐标点找到对应的网格ID（使用最近邻算法）
    
    Args:
        x: 点的x坐标
        y: 点的y坐标
        ne_data: 网格数据
        
    Returns:
        int | None: 对应的网格ID，如果没找到则返回None
    """
    
    min_distance = float('inf')
    nearest_grid_id = None
    
    # 遍历所有网格，找到距离最近的网格中心点
    for i in range(len(ne_data.xe_list)):
        grid_x = ne_data.xe_list[i]
        grid_y = ne_data.ye_list[i]
        
        # 计算欧几里得距离
        distance = math.sqrt((x - grid_x)**2 + (y - grid_y)**2)
        
        if distance < min_distance:
            min_distance = distance
            nearest_grid_id = ne_data.grid_id_list[i]
    
    return nearest_grid_id

def find_grid_for_feature_point(feature_json: dict, ne_data: NeData, grid_result: np.ndarray = None) -> list[int]:
    """
    根据GeoJSON格式的点要素找到对应的网格ID列表
    
    Args:
        feature_json: GeoJSON格式的地理要素（Feature或FeatureCollection）
        ne_data: 网格数据
        grid_result: 网格数据数组，每行包含 [网格ID, 中心x坐标, 中心y坐标, 半边长]
        
    Returns:
        list[int]: 与点要素对应的网格ID列表
    """
    if not feature_json:
        return []
    
    grid_ids = []
    
    # 检查是否是FeatureCollection
    if feature_json.get('type') == 'FeatureCollection':
        features = feature_json.get('features', [])
        
        # 处理FeatureCollection中的每个Feature
        for feature in features:
            grid_ids.extend(find_grid_for_feature_point(feature, ne_data, grid_result))
            
        # 去重
        return list(set(grid_ids))
    
    # 处理单个Feature
    if 'geometry' not in feature_json:
        return []
    
    geometry = feature_json['geometry']
    geom_type = geometry.get('type', '').lower()
    coordinates = geometry.get('coordinates', [])
    
    if geom_type == 'point':
        # 对于Point，找到对应的网格ID
        if len(coordinates) >= 2:
            x, y = coordinates[0], coordinates[1]
            if grid_result is not None:
                # 使用grid_result查找点所在的网格
                grid_id = find_grid_for_point_using_grid_result(x, y, grid_result)
                if grid_id is not None:
                    grid_ids.append(grid_id)
                    logger.info(f"点坐标 ({x}, {y}) 使用grid_result对应网格ID: {grid_id}")
                else:
                    # 如果使用grid_result找不到，回退到使用ne_data
                    grid_id = find_grid_for_point(x, y, ne_data)
                    if grid_id is not None:
                        grid_ids.append(grid_id)
                        logger.info(f"点坐标 ({x}, {y}) 回退使用ne_data对应网格ID: {grid_id}")
            else:
                # 如果没有提供grid_result，使用ne_data
                grid_id = find_grid_for_point(x, y, ne_data)
                if grid_id is not None:
                    grid_ids.append(grid_id)
                    logger.info(f"点坐标 ({x}, {y}) 对应网格ID: {grid_id}")
    
    elif geom_type == 'multipoint':
        # 对于MultiPoint，处理每个点
        for point_coords in coordinates:
            if len(point_coords) >= 2:
                x, y = point_coords[0], point_coords[1]
                if grid_result is not None:
                    # 使用grid_result查找点所在的网格
                    grid_id = find_grid_for_point_using_grid_result(x, y, grid_result)
                    if grid_id is not None:
                        grid_ids.append(grid_id)
                        logger.info(f"多点坐标 ({x}, {y}) 使用grid_result对应网格ID: {grid_id}")
                    else:
                        # 如果使用grid_result找不到，回退到使用ne_data
                        grid_id = find_grid_for_point(x, y, ne_data)
                        if grid_id is not None:
                            grid_ids.append(grid_id)
                            logger.info(f"多点坐标 ({x}, {y}) 回退使用ne_data对应网格ID: {grid_id}")
                else:
                    # 如果没有提供grid_result，使用ne_data
                    grid_id = find_grid_for_point(x, y, ne_data)
                    if grid_id is not None:
                        grid_ids.append(grid_id)
                        logger.info(f"多点坐标 ({x}, {y}) 对应网格ID: {grid_id}")
    
    return grid_ids

def find_grid_for_point_using_grid_result(x: float, y: float, grid_result: np.ndarray) -> int | None:
    """
    使用grid_result查找点所在的网格ID
    
    Args:
        x: 点的x坐标
        y: 点的y坐标
        grid_result: 网格数据数组，每行包含 [网格ID, 中心x坐标, 中心y坐标, 半边长]
        
    Returns:
        int | None: 对应的网格ID，如果没找到则返回None
    """
    if grid_result is None or len(grid_result) == 0:
        return None
    
    for grid_row in grid_result:
        if len(grid_row) < 4:
            continue
        
        grid_id = int(grid_row[0])
        grid_center_x = float(grid_row[1])
        grid_center_y = float(grid_row[2])
        half_size = float(grid_row[3])
        
        # 计算网格的边界
        min_x = grid_center_x - half_size
        max_x = grid_center_x + half_size
        min_y = grid_center_y - half_size
        max_y = grid_center_y + half_size
        
        # 检查点是否在网格内
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return grid_id
    
    return None

def apply_add_fence_action(fence_params: FenceParams, model_data: dict) -> dict:

    logger.info("开始应用基围")
    elevation_delta = fence_params.elevation_delta
    landuse_type = fence_params.landuse_type
    feature_json = transform_feature_4326_to_2326(fence_params.feature)

    ne_data: NeData = model_data.get('ne', {})
    ns_data: NsData = model_data.get('ns', {})
    
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

def apply_add_gate_action(gate_params: GateParams, model_data: dict, grid_result: np.ndarray) -> dict:

    logger.info("开始应用闸门")
    up_stream = gate_params.up_stream
    down_stream = gate_params.down_stream
    gate_height = gate_params.gate_height
    feature_json = transform_feature_4326_to_2326(gate_params.feature)

    print(f"闸门信息: {model_data['gate']}")

    # 获取与闸门相交的网格号
    grid_ids = get_grids_intersecting_with_line(feature_json, grid_result)
    logger.info(f"与闸门相交的网格号: {grid_ids}")
    
    # 处理上游点
    up_stream_grid_id = up_stream
    transformed_up_stream = transform_point_list_4326_to_2326(up_stream)
    # 优先使用grid_result查找
    up_grid_id = find_grid_for_point_using_grid_result(transformed_up_stream[0], transformed_up_stream[1], grid_result)
    if up_grid_id is not None:
        up_stream_grid_id = up_grid_id
        logger.info(f"上游坐标点 {up_stream} (4326) -> {transformed_up_stream} (2326) 对应网格ID: {up_grid_id}")
    
    # 处理下游点
    down_stream_grid_id = down_stream
    transformed_down_stream = transform_point_list_4326_to_2326(down_stream)
    # 优先使用grid_result查找
    down_grid_id = find_grid_for_point_using_grid_result(transformed_down_stream[0], transformed_down_stream[1], grid_result)
    if down_grid_id is not None:
        down_stream_grid_id = down_grid_id
        logger.info(f"下游坐标点 {down_stream} (4326) -> {transformed_down_stream} (2326) 对应网格ID: {down_grid_id}")

    gate_data: Gate = model_data.get('gate')
    gate_data.ud_stream_list.append(up_stream_grid_id)
    gate_data.ud_stream_list.append(down_stream_grid_id)
    gate_data.gate_height_list.append(gate_height)
    gate_data.grid_id_list.append(grid_ids)



    model_data['gate'] = gate_data

    print(f"更新后的闸门信息: {model_data['gate']}")
    
    return model_data

def get_grids_intersecting_with_line(feature_json: dict, grid_result: np.ndarray) -> list:
    """
    获取与线要素相交的网格ID列表
    
    Args:
        feature_json: GeoJSON格式的地理要素（已转换为EPSG:2326），可以是Feature或FeatureCollection
        grid_result: 网格数据数组，每行包含 [网格ID, 中心x坐标, 中心y坐标, 半边长]
        
    Returns:
        list: 与线要素相交的网格ID列表
    """
    if not feature_json:
        return []
    
    # 检查是否是FeatureCollection
    if feature_json.get('type') == 'FeatureCollection':
        print("处理FeatureCollection中的多个Feature")
        features = feature_json.get('features', [])
        all_intersecting_grid_ids = []
        
        # 处理FeatureCollection中的每个Feature
        for feature in features:
            intersecting_grid_ids = get_grids_intersecting_with_line(feature, grid_result)
            all_intersecting_grid_ids.extend(intersecting_grid_ids)
        
        # 去重
        return list(set(all_intersecting_grid_ids))
    
    # 处理单个Feature
    if 'geometry' not in feature_json:
        return []
    
    geometry = feature_json['geometry']
    geom_type = geometry.get('type', '').lower()
    coordinates = geometry.get('coordinates', [])
    
    # 只处理LineString或MultiLineString几何类型
    if geom_type != 'linestring' and geom_type != 'multilinestring':
        logger.warning(f"几何类型 {geom_type} 不是线要素，无法计算相交的网格")
        return []
    
    intersecting_grid_ids = []
    
    # 处理所有线段
    line_segments = []
    if geom_type == 'linestring':
        # 单线，将所有相邻点对构成线段
        print(len(coordinates)-1)
        for i in range(len(coordinates) - 1):
            line_segments.append((coordinates[i], coordinates[i + 1]))
    elif geom_type == 'multilinestring':
        # 多线，每条线都需要处理
        for line in coordinates:
            for i in range(len(line) - 1):
                line_segments.append((line[i], line[i + 1]))
    
    # 遍历所有网格，检查是否与任何线段相交
    for grid_row in grid_result:
        if len(grid_row) < 4:
            continue
        
        grid_id = int(grid_row[0])
        grid_center_x = float(grid_row[1])
        grid_center_y = float(grid_row[2])
        half_size = float(grid_row[3])
        
        # 计算网格的四个顶点
        min_x = grid_center_x - half_size
        max_x = grid_center_x + half_size
        min_y = grid_center_y - half_size
        max_y = grid_center_y + half_size
        
        # 网格的四条边
        grid_edges = [
            ((min_x, min_y), (max_x, min_y)),  # 下边
            ((max_x, min_y), (max_x, max_y)),  # 右边
            ((max_x, max_y), (min_x, max_y)),  # 上边
            ((min_x, max_y), (min_x, min_y))   # 左边
        ]
        
        # 检查线段是否与网格相交
        for line_segment in line_segments:
            line_p1, line_p2 = line_segment
            line_x1, line_y1 = line_p1
            line_x2, line_y2 = line_p2
            
            # 检查线段是否完全在网格外部
            if (max(line_x1, line_x2) < min_x or
                min(line_x1, line_x2) > max_x or
                max(line_y1, line_y2) < min_y or
                min(line_y1, line_y2) > max_y):
                continue
            
            # 检查线段端点是否在网格内部
            if (min_x <= line_x1 <= max_x and min_y <= line_y1 <= max_y) or \
               (min_x <= line_x2 <= max_x and min_y <= line_y2 <= max_y):
                intersecting_grid_ids.append(grid_id)
                break
            
            # 检查线段是否与网格边界相交
            for grid_edge in grid_edges:
                grid_p1, grid_p2 = grid_edge
                grid_x1, grid_y1 = grid_p1
                grid_x2, grid_y2 = grid_p2
                
                if do_line_segments_intersect(
                    line_x1, line_y1, line_x2, line_y2,
                    grid_x1, grid_y1, grid_x2, grid_y2
                ):
                    intersecting_grid_ids.append(grid_id)
                    break
            else:
                # 如果与网格的所有边都不相交，继续检查下一个线段
                continue
            
            # 如果已找到相交，跳出线段循环
            break
    
    return intersecting_grid_ids

def do_line_segments_intersect(x1: float, y1: float, x2: float, y2: float, 
                              x3: float, y3: float, x4: float, y4: float) -> bool:
    """
    检查两条线段是否相交
    
    Args:
        x1, y1: 第一条线段的起点
        x2, y2: 第一条线段的终点
        x3, y3: 第二条线段的起点
        x4, y4: 第二条线段的终点
        
    Returns:
        bool: True表示线段相交，False表示不相交
    """
    # 计算方向
    def direction(x1, y1, x2, y2, x3, y3):
        return (x3 - x1) * (y2 - y1) - (x2 - x1) * (y3 - y1)
    
    # 检查点是否在线段上
    def on_segment(x1, y1, x2, y2, x3, y3):
        return (min(x1, x2) <= x3 <= max(x1, x2) and 
                min(y1, y2) <= y3 <= max(y1, y2))
    
    # 计算方向值
    d1 = direction(x3, y3, x4, y4, x1, y1)
    d2 = direction(x3, y3, x4, y4, x2, y2)
    d3 = direction(x1, y1, x2, y2, x3, y3)
    d4 = direction(x1, y1, x2, y2, x4, y4)
    
    # 线段相交的一般情况
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # 处理共线或端点在另一条线段上的情况
    if d1 == 0 and on_segment(x3, y3, x4, y4, x1, y1):
        return True
    if d2 == 0 and on_segment(x3, y3, x4, y4, x2, y2):
        return True
    if d3 == 0 and on_segment(x1, y1, x2, y2, x3, y3):
        return True
    if d4 == 0 and on_segment(x1, y1, x2, y2, x4, y4):
        return True
    
    return False

def apply_transfer_water_action(transfer_water_params: TransferWaterParams, model_data: dict, watergroups: list, grid_result: np.ndarray = None) -> list:

    logger.info("开始应用调水")
    ne_data: NeData = model_data.get('ne', {})
    
    # 处理调水源头
    from_grid = transfer_water_params.from_grid
    if isinstance(from_grid, dict):  # 如果是GeoJSON Feature或FeatureCollection
        # 转换坐标系
        from_grid_feature = transform_feature_4326_to_2326(from_grid)
        # 查找对应的网格ID（优先使用grid_result）
        from_grid_ids = find_grid_for_feature_point(from_grid_feature, ne_data, grid_result)
        if from_grid_ids:
            from_grid = from_grid_ids[0]  # 取第一个匹配的网格ID
            logger.info(f"调水源头Feature点对应网格ID: {from_grid}")
    elif isinstance(from_grid, list) and len(from_grid) >= 2:  # 如果是坐标点 [lon, lat]
        transformed_from_grid = transform_point_list_4326_to_2326(from_grid)
        # 优先使用grid_result查找
        from_grid_id = None
        if grid_result is not None:
            from_grid_id = find_grid_for_point_using_grid_result(transformed_from_grid[0], transformed_from_grid[1], grid_result)
        if from_grid_id is None:  # 如果使用grid_result找不到，回退到使用ne_data
            from_grid_id = find_grid_for_point(transformed_from_grid[0], transformed_from_grid[1], ne_data)
        if from_grid_id is not None:
            from_grid = from_grid_id
            logger.info(f"调水源头坐标点 {from_grid} (4326) -> {transformed_from_grid} (2326) 对应网格ID: {from_grid}")
    
    # 处理调水终点
    to_grid = transfer_water_params.to_grid
    if isinstance(to_grid, dict):  # 如果是GeoJSON Feature或FeatureCollection
        # 转换坐标系
        to_grid_feature = transform_feature_4326_to_2326(to_grid)
        # 查找对应的网格ID（优先使用grid_result）
        to_grid_ids = find_grid_for_feature_point(to_grid_feature, ne_data, grid_result)
        if to_grid_ids:
            to_grid = to_grid_ids[0]  # 取第一个匹配的网格ID
            logger.info(f"调水终点Feature点对应网格ID: {to_grid}")
    elif isinstance(to_grid, list) and len(to_grid) >= 2:  # 如果是坐标点 [lon, lat]
        transformed_to_grid = transform_point_list_4326_to_2326(to_grid)
        # 优先使用grid_result查找
        to_grid_id = None
        if grid_result is not None:
            to_grid_id = find_grid_for_point_using_grid_result(transformed_to_grid[0], transformed_to_grid[1], grid_result)
        if to_grid_id is None:  # 如果使用grid_result找不到，回退到使用ne_data
            to_grid_id = find_grid_for_point(transformed_to_grid[0], transformed_to_grid[1], ne_data)
        if to_grid_id is not None:
            to_grid = to_grid_id
            logger.info(f"调水终点坐标点 {to_grid} (4326) -> {transformed_to_grid} (2326) 对应网格ID: {to_grid}")
    
    watergroup = {
        'from_grid': from_grid,
        'to_grid': to_grid,
        'q': transfer_water_params.q
    }
    watergroups.append(watergroup)
    return watergroups
