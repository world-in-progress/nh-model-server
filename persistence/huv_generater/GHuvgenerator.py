import os
import time
import numpy as np
from osgeo import gdal, osr, ogr
from shapely.geometry import Polygon
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
from .dataset2huv import process_huv_to_image_from_datasets

def solve_irregular_grid(ne_file_path, ns_file_path):
    """
    解算不规则矢量格网 - 优化版本，使用NumPy和多线程
    
    参数:
        ne_file_path: ne.txt文件路径
        ns_file_path: ns.txt文件路径
    
    返回:
        tuple: (result_array, header, bbox)
        - result_array: numpy数组，包含：网格id，网格xy坐标，网格二分之一边长
          格式：[网格id, x坐标, y坐标, 半边长]
        - header: 列名列表
        - bbox: 包围盒字典，包含 min_x, max_x, min_y, max_y
    """
    header = ['id', 'x', 'y', 'length']
    
    def read_ne_file(file_path):
        """优化的ne文件读取"""
        print("正在读取ne文件...")
        try:
            # 使用pandas风格的快速读取，但用numpy实现
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # 预分配数组
            data = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 6:  # 确保有足够的列
                        try:
                            grid_id = int(parts[0])
                            left_edge_id = int(parts[5])
                            grid_x = float(parts[-4])
                            grid_y = float(parts[-3])
                            data.append([grid_id, left_edge_id, grid_x, grid_y])
                        except (ValueError, IndexError):
                            continue
            
            return np.array(data, dtype=np.float64)
        except Exception as e:
            print(f"读取ne文件出错: {e}")
            return np.array([])
    
    def read_ns_file(file_path):
        """优化的ns文件读取"""
        print("正在读取ns文件...")
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # 使用字典推导式快速建立映射
            ns_data = {}
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 4:  # 确保有足够的列
                        try:
                            edge_id = int(parts[0])
                            edge_x = float(parts[-4])
                            edge_y = float(parts[-3])
                            ns_data[edge_id] = [edge_x, edge_y]
                        except (ValueError, IndexError):
                            continue
            
            return ns_data
        except Exception as e:
            print(f"读取ns文件出错: {e}")
            return {}
    
    # 使用多线程并行读取文件
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_ne = executor.submit(read_ne_file, ne_file_path)
        future_ns = executor.submit(read_ns_file, ns_file_path)
        
        ne_data = future_ne.result()
        ns_data = future_ns.result()
    
    if len(ne_data) == 0:
        print("警告：ne文件未读取到有效数据")
        return np.array([]), header, {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
    
    print(f"读取了 {len(ne_data)} 个网格")
    print(f"读取了 {len(ns_data)} 条边")
    
    # 向量化计算网格数据
    print("正在解算网格...")
    
    # 提取数据列
    grid_ids = ne_data[:, 0].astype(int)
    left_edge_ids = ne_data[:, 1].astype(int)
    grid_xs = ne_data[:, 2]
    grid_ys = ne_data[:, 3]
    
    # 创建结果数组
    result_grids = []
    
    # 向量化查找对应边
    for i in range(len(ne_data)):
        grid_id = grid_ids[i]
        left_edge_id = left_edge_ids[i]
        grid_x = grid_xs[i]
        grid_y = grid_ys[i]
        
        if left_edge_id in ns_data:
            edge_x, edge_y = ns_data[left_edge_id]
            x_diff = edge_x - grid_x
            half_edge_length = int(abs(x_diff) + 0.5)  # 四舍五入
            result_grids.append([grid_id, grid_x, grid_y, half_edge_length])
        else:
            result_grids.append([grid_id, grid_x, grid_y, 0])
    
    print(f"解算完成，生成了 {len(result_grids)} 个新网格")
    
    # 计算格网包围盒范围（向量化计算）
    print("正在计算格网包围盒...")
    if ns_data:
        edge_coordinates = np.array(list(ns_data.values()))
        bbox = {
            'min_x': np.min(edge_coordinates[:, 0]),
            'max_x': np.max(edge_coordinates[:, 0]),
            'min_y': np.min(edge_coordinates[:, 1]),
            'max_y': np.max(edge_coordinates[:, 1])
        }
        print(f"包围盒范围: X[{bbox['min_x']:.3f}, {bbox['max_x']:.3f}], Y[{bbox['min_y']:.3f}, {bbox['max_y']:.3f}]")
    else:
        bbox = {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
        print("警告：未找到边界数据，使用默认包围盒")
    
    # 转换为numpy数组
    result_array = np.array(result_grids, dtype=np.float64)
    
    return result_array, header, bbox


def merge_with_result_data(grid_result, result_file_path):
    """
    将grid_result与result.dat文件数据合并 - 高度优化版本，完全向量化处理
    
    参数:
        grid_result: solve_irregular_grid函数返回的网格数据数组
        result_file_path: result.dat文件路径
    
    返回:
        tuple: (merged_array, merged_header)
        merged_array格式：[网格id, x坐标, y坐标, 半边长, u, v, h]
    """
    print("正在读取result.dat文件...")
    
    # 使用numpy快速读取result.dat文件
    try:
        result_data = np.genfromtxt(result_file_path, dtype=np.float64, skip_header=1, delimiter="\t")
        print(f"读取了 {len(result_data)} 行结果数据")
    except Exception as e:
        print(f"读取result.dat文件失败: {e}")
        return np.array([]), []
    
    if len(result_data) == 0:
        print("警告：result.dat文件为空")
        return np.array([]), []
    
    # 向量化处理result数据
    grid_ids_result = result_data[:, 0].astype(int)  # 第1列：grid_id
    h_values = result_data[:, 1]                     # 第2列：h
    u_values = result_data[:, 2]                     # 第3列：u  
    v_values = result_data[:, 3]                     # 第4列：v
    ze_values = result_data[:, 4]                    # 第5列：ze
    
    # 向量化计算h值：h = max(ze, 0)
    calculated_h = np.maximum(h_values-ze_values, 0)
    
    # 创建结果数据查找表（使用scipy.sparse可能更快，但这里用numpy实现）
    # 为了加速查找，我们先对结果数据按grid_id排序
    sort_indices = np.argsort(grid_ids_result)
    sorted_grid_ids = grid_ids_result[sort_indices]
    sorted_u_values = u_values[sort_indices]
    sorted_v_values = v_values[sort_indices]
    sorted_h_values = calculated_h[sort_indices]
    
    print(f"建立了 {len(sorted_grid_ids)} 个网格的结果映射")
    
    # 提取grid_result中的数据
    grid_ids_input = grid_result[:, 0].astype(int)
    n_grids = len(grid_result)
    
    # 预分配结果数组
    merged_data = np.zeros((n_grids, 7), dtype=np.float64)
    
    # 填充基础数据（ID、坐标、半边长）
    merged_data[:, 0] = grid_ids_input      # grid_id
    merged_data[:, 1] = grid_result[:, 1]   # x坐标
    merged_data[:, 2] = grid_result[:, 2]   # y坐标
    merged_data[:, 3] = grid_result[:, 3]   # 半边长
    
    # 使用numpy.searchsorted进行高效查找
    # 这比字典查找或循环快得多
    search_indices = np.searchsorted(sorted_grid_ids, grid_ids_input)
    
    # 处理边界情况，确保索引不越界
    valid_indices = (search_indices < len(sorted_grid_ids)) & \
                   (sorted_grid_ids[search_indices] == grid_ids_input)
    
    # 向量化填充UVH数据
    matched_indices = search_indices[valid_indices]
    valid_grid_positions = np.where(valid_indices)[0]
    
    merged_data[valid_grid_positions, 4] = sorted_u_values[matched_indices]  # u
    merged_data[valid_grid_positions, 5] = sorted_v_values[matched_indices]  # v
    merged_data[valid_grid_positions, 6] = sorted_h_values[matched_indices]  # h
    
    matched_count = np.sum(valid_indices)
    unmatched_count = n_grids - matched_count
    
    print(f"数据匹配完成：匹配 {matched_count} 个，未匹配 {unmatched_count} 个")
    
    # 更新表头
    merged_header = ['id', 'x', 'y', 'length', 'u', 'v', 'h']
    
    # 输出统计信息
    if len(merged_data) > 0:
        print(f"合并后数据统计:")
        u_col = merged_data[:, 4]
        v_col = merged_data[:, 5]
        h_col = merged_data[:, 6]
        
        print(f"  U速度范围: [{u_col.min():.3f}, {u_col.max():.3f}]")
        print(f"  V速度范围: [{v_col.min():.3f}, {v_col.max():.3f}]")
        print(f"  H值范围: [{h_col.min():.3f}, {h_col.max():.3f}]")
        print(f"  非零H值点数: {np.sum(h_col > 0)}/{len(merged_data)}")
    
    return merged_data, merged_header

def grid_edge(merged_result):
    """
    计算多个格网组成的不规则多边形外边界。
    使用缓冲区处理来消除悬挂线。
    """
    from shapely.geometry import box
    from shapely.ops import unary_union
    
    grid_x = merged_result[:, 1]
    grid_y = merged_result[:, 2]
    half_len = merged_result[:, 3]

    # 构造所有正方形 Polygon
    polygons = [
        box(x - l, y - l, x + l, y + l)
        for x, y, l in zip(grid_x, grid_y, half_len)
    ]

    # 合并为一个大多边形
    merged_polygon = unary_union(polygons)

    # 如果结果是 MultiPolygon，只取最大的
    if merged_polygon.geom_type == 'MultiPolygon':
        merged_polygon = max(merged_polygon.geoms, key=lambda p: p.area)

    # 使用小的缓冲区来平滑边界，消除悬挂线
    # 先向外扩展一点，再向内收缩相同距离
    buffer_distance = min(half_len) * 0.01  # 使用最小格网尺寸的1%作为缓冲距离
    smoothed_polygon = merged_polygon.buffer(buffer_distance).buffer(-buffer_distance)
    
    # 如果缓冲后又变成MultiPolygon，再次取最大的
    if smoothed_polygon.geom_type == 'MultiPolygon':
        smoothed_polygon = max(smoothed_polygon.geoms, key=lambda p: p.area)

    # 提取外边界坐标
    boundary_points = list(smoothed_polygon.exterior.coords)

    return boundary_points

def save_boundary_to_shp_osgeo(boundary_points, output_path, epsg=2326):
    """
    使用 osgeo（GDAL/OGR）将边界 Polygon 存储为 .shp 文件，坐标系为 EPSG:2326

    参数:
        boundary_points: List[(x, y)]，边界坐标列表
        output_path: 输出 shapefile 文件路径，例如 "output/boundary.shp"
        epsg: 投影坐标系 EPSG 代码，默认 2326
    """
    # 创建 shapely Polygon，自动闭合
    polygon = Polygon(boundary_points)

    # 删除已有文件（Shapefile 有多个相关文件）
    driver = ogr.GetDriverByName("ESRI Shapefile")
    driver.DeleteDataSource(output_path)

    # 创建 shapefile 数据源
    datasource = driver.CreateDataSource(output_path)

    # 设置空间参考
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(epsg)

    # 创建图层
    layer = datasource.CreateLayer("boundary", spatial_ref, ogr.wkbPolygon)

    # 添加属性字段（ID）
    field = ogr.FieldDefn("ID", ogr.OFTInteger)
    layer.CreateField(field)

    # 构建 shapely -> ogr 的 Polygon 几何
    ogr_polygon = ogr.CreateGeometryFromWkt(polygon.wkt)

    # 创建 feature 并写入图层
    feature_def = layer.GetLayerDefn()
    feature = ogr.Feature(feature_def)
    feature.SetField("ID", 1)
    feature.SetGeometry(ogr_polygon)
    layer.CreateFeature(feature)

    # 清理
    feature = None
    datasource = None

def create_tiled_datasets(merged_result, no_data_value=-9999, 
                         source_epsg=2326, bbox=None, pixel=None):
    
    print("开始处理网格数据...")
    
    # 提取数据

    x_coords = merged_result[:, 1]
    y_coords = merged_result[:, 2] 
    lengths = merged_result[:, 3]
    
    print(f"处理 {len(merged_result)} 个网格")
    
    # 计算分辨率（使用最小的length值）
    min_length = np.min(lengths[lengths > 0])
    if pixel:
        pixel_size = pixel
    else:
        pixel_size = min_length  # 使用最小网格半边长作为像素大小（单位：米）
    
    print(f"网格半边长范围: [{np.min(lengths):.2f}, {np.max(lengths):.2f}] 米")
    print(f"选择的像素大小: {pixel_size:.2f} 米")
    
    # 计算数据范围
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    
    print(f"数据范围: X[{min_x:.6f}, {max_x:.6f}], Y[{min_y:.6f}, {max_y:.6f}]")
    
    # 使用bbox计算总的栅格尺寸
    if bbox is not None:
        bbox_min_x, bbox_max_x = bbox['min_x'], bbox['max_x']
        bbox_min_y, bbox_max_y = bbox['min_y'], bbox['max_y']
    else:
        # 如果没有提供bbox，使用数据范围
        bbox_min_x, bbox_max_x = min_x, max_x
        bbox_min_y, bbox_max_y = min_y, max_y
    
    total_width = int(np.ceil((bbox_max_x - bbox_min_x) / pixel_size))
    total_height = int(np.ceil((bbox_max_y - bbox_min_y) / pixel_size))
    
    print(f"使用包围盒范围: X[{bbox_min_x:.6f}, {bbox_max_x:.6f}], Y[{bbox_min_y:.6f}, {bbox_max_y:.6f}]")
    print(f"栅格尺寸: {total_width} x {total_height} 像素")
    
    # 创建空的三通道GeoTIFF数据集
    print("正在创建GDAL数据集...")
    
    # 使用内存驱动创建数据集
    driver = gdal.GetDriverByName('MEM')
    if driver is None:
        raise RuntimeError("无法获取GDAL内存驱动")
    
    # 创建三通道数据集 (U, V, H)
    dataset = driver.Create('', total_width, total_height, 3, gdal.GDT_Float32)
    if dataset is None:
        raise RuntimeError("无法创建GDAL数据集")
    
    # 设置地理变换参数
    # GeoTransform = [左上角X, 像素宽度, 0, 左上角Y, 0, -像素高度]
    geotransform = [
        bbox_min_x,     # 左上角X坐标
        pixel_size,     # 像素宽度（米）
        0,              # 旋转参数（通常为0）
        bbox_max_y,     # 左上角Y坐标
        0,              # 旋转参数（通常为0）
        -pixel_size     # 像素高度（负值，因为Y轴向下）
    ]
    dataset.SetGeoTransform(geotransform)
    
    # 设置坐标系
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(source_epsg)
    dataset.SetProjection(srs.ExportToWkt())
    
    # 初始化所有波段为无数据值
    for band_idx in range(1, 4):  # 1, 2, 3 对应 U, V, H
        band = dataset.GetRasterBand(band_idx)
        band.SetNoDataValue(float(no_data_value))  # 确保是 float 类型
        # 填充整个波段为无数据值
        band.Fill(float(no_data_value))
        
        # 设置波段描述
        if band_idx == 1:
            band.SetDescription('U_velocity')
        elif band_idx == 2:
            band.SetDescription('V_velocity')
        elif band_idx == 3:
            band.SetDescription('H_depth')
    
    print(f"成功创建 {total_width}x{total_height} 像素的三通道数据集")
    
    # 填充网格数据到数据集 - 优化版本
    print("正在填充网格数据...")
    
    # 创建完整的数据数组
    u_array = np.full((total_height, total_width), float(no_data_value), dtype=np.float32)
    v_array = np.full((total_height, total_width), float(no_data_value), dtype=np.float32)
    h_array = np.full((total_height, total_width), float(no_data_value), dtype=np.float32)
    
    # 过滤有效数据
    valid_mask = merged_result[:, 3] > 0  # half_length > 0
    valid_data = merged_result[valid_mask]
    
    if len(valid_data) > 0:
        # 向量化计算所有网格的像素坐标
        grid_x = valid_data[:, 1]
        grid_y = valid_data[:, 2]
        half_length = valid_data[:, 3]
        u_vals = valid_data[:, 4].astype(np.float32)
        v_vals = valid_data[:, 5].astype(np.float32)
        h_vals = valid_data[:, 6].astype(np.float32)
        
        # 计算格子左上角坐标
        grid_left = grid_x - half_length
        grid_top = grid_y + half_length
        
        # 计算相对位置
        rel_x = grid_left - bbox_min_x
        rel_y = bbox_max_y - grid_top
        
        # 向量化计算像素坐标
        pixel_col = np.where(rel_x >= 0, 
                           (rel_x / pixel_size + 0.5).astype(int),
                           -((-rel_x) / pixel_size + 0.5).astype(int))
        pixel_row = np.where(rel_y >= 0,
                           (rel_y / pixel_size + 0.5).astype(int), 
                           -((-rel_y) / pixel_size + 0.5).astype(int))
        
        # 计算像素范围
        full_length = half_length * 2
        pixels_per_grid = (full_length / pixel_size + 0.5).astype(int)
        
        # 批量处理网格
        filled_pixels = 0
        print(f"使用向量化方法处理 {len(valid_data)} 个有效网格...")
        
        for i in tqdm(range(len(valid_data)), desc="填充网格数据"):
            start_col = max(0, pixel_col[i])
            end_col = min(total_width, pixel_col[i] + pixels_per_grid[i])
            start_row = max(0, pixel_row[i])
            end_row = min(total_height, pixel_row[i] + pixels_per_grid[i])
            
            if start_col < end_col and start_row < end_row:
                # 直接在数组中设置值
                u_array[start_row:end_row, start_col:end_col] = u_vals[i]
                v_array[start_row:end_row, start_col:end_col] = v_vals[i] 
                h_array[start_row:end_row, start_col:end_col] = h_vals[i]
                filled_pixels += (end_row - start_row) * (end_col - start_col)
    
    # 一次性写入所有波段数据
    print("正在写入波段数据...")
    u_band = dataset.GetRasterBand(1)
    v_band = dataset.GetRasterBand(2)
    h_band = dataset.GetRasterBand(3)
    
    u_band.WriteArray(u_array)
    v_band.WriteArray(v_array)
    h_band.WriteArray(h_array)
    
    # 刷新缓存
    dataset.FlushCache()
    
    print(f"数据填充完成！")
    print(f"  处理了 {len(valid_data)} 个有效网格")
    print(f"  填充了 {filled_pixels} 个像素")
    print(f"  填充率: {filled_pixels / (total_width * total_height) * 100:.2f}%")

    return dataset, pixel_size

def downsample_dataset(dataset, pixel_size, target_resolution=20, no_data_value=-9999):
    """
    对GDAL数据集进行下采样，从原始分辨率下采样到指定分辨率
    使用向量化操作和多线程优化性能
    
    参数:
        dataset: 输入的GDAL数据集
        pixel_size: 原始像素大小（米）
        target_resolution: 目标分辨率（米），默认20米
        no_data_value: 无数据值，默认-9999
    
    返回:
        下采样后的GDAL数据集
        #设定降采样参数
        width = 2
        height = 2

        #将图像降采样
        ds_gray1 = np.mean(gray.reshape(-1, height, gray.shape[1]), axis=1) 
        ds_gray = np.mean(ds_gray1.reshape(-1, width, ds_gray1.shape[-1]), axis=1)
    """
    print(f"开始下采样，目标分辨率: {target_resolution}米...")
    
    # 获取原始数据集信息
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    original_width = dataset.RasterXSize
    original_height = dataset.RasterYSize
    band_count = dataset.RasterCount
    
    # 使用传入的像素大小
    original_pixel_size = pixel_size
    
    print(f"原始数据集: {original_width}x{original_height}, 像素大小: {original_pixel_size}米")
    if original_pixel_size < target_resolution:
        # 计算采样窗口大小
        sample_window_size = int(target_resolution / original_pixel_size)
    else:
        sample_window_size = 1
    
    print(f"采样窗口大小: {sample_window_size}x{sample_window_size}")
    
    # 计算新的数据集尺寸 - 基于扩展后的尺寸而不是原始尺寸
    # 确保能完整覆盖原始数据，向上取整
    new_width = (original_width + sample_window_size - 1) // sample_window_size
    new_height = (original_height + sample_window_size - 1) // sample_window_size
    
    print(f"下采样后尺寸: {new_width}x{new_height}")
    
    # 创建新的数据集
    driver = gdal.GetDriverByName('MEM')
    downsampled_dataset = driver.Create('', new_width, new_height, band_count, gdal.GDT_Float32)
    
    # 设置新的地理变换参数
    new_geotransform = [
        geotransform[0],  # 左上角X坐标保持不变
        target_resolution,  # 新的像素宽度
        geotransform[2],  # 旋转参数
        geotransform[3],  # 左上角Y坐标保持不变
        geotransform[4],  # 旋转参数
        -target_resolution  # 新的像素高度（负值）
    ]
    downsampled_dataset.SetGeoTransform(new_geotransform)
    downsampled_dataset.SetProjection(projection)
    
    def process_band_optimized(band_idx):
        """优化的波段处理函数，使用向量化操作，对除不尽的部分进行nodata填充"""
        # 获取原始波段
        original_band = dataset.GetRasterBand(band_idx)
        original_data = original_band.ReadAsArray().astype(np.float32)
        
        # 创建新的波段
        new_band = downsampled_dataset.GetRasterBand(band_idx)
        new_band.SetNoDataValue(float(no_data_value))  # 确保是 float 类型
        new_band.SetDescription(original_band.GetDescription())
        
        # 计算需要扩展到的尺寸（能被窗口大小整除）
        padded_height = new_height * sample_window_size
        padded_width = new_width * sample_window_size
        
        # 创建扩展后的数组，用nodata值填充
        padded_data = np.full((padded_height, padded_width), no_data_value, dtype=np.float32)
        
        # 将原始数据复制到扩展数组的左上角
        padded_data[:original_height, :original_width] = original_data
        
        print(f"波段 {band_idx}: 原始尺寸 {original_height}x{original_width}, 扩展后尺寸 {padded_height}x{padded_width}")
        
        # 重塑数组以便于批量处理
        # 形状: (new_height, sample_window_size, new_width, sample_window_size)
        reshaped = padded_data.reshape(
            new_height, sample_window_size,
            new_width, sample_window_size
        )
        
        # 转换为 (new_height, new_width, sample_window_size, sample_window_size)
        reshaped = reshaped.transpose(0, 2, 1, 3)
        
        # 重塑为 (new_height, new_width, window_size^2)
        window_data = reshaped.reshape(new_height, new_width, -1)
        
        # 创建掩码标识有效数据
        valid_mask = window_data != no_data_value
        
        # 计算每个窗口的有效数据数量
        valid_counts = np.sum(valid_mask, axis=2)
        
        # 使用掩码数组计算平均值
        window_data_masked = np.where(valid_mask, window_data, 0)
        window_sums = np.sum(window_data_masked, axis=2)
        
        # 计算平均值，需要至少 50% 有效像素才参与平均
        new_data = np.full((new_height, new_width), no_data_value, dtype=np.float32)
        
        # 需要至少 50% 有效像素才参与平均
        min_valid_ratio = 0.5
        window_size_total = sample_window_size * sample_window_size
        threshold = min_valid_ratio * window_size_total  # 计算有效像素的阈值（50%）
        has_valid = valid_counts >= threshold  # 只有有效像素数量达到阈值的窗口才参与计算
        
        # 对满足条件的窗口计算平均值
        new_data[has_valid] = window_sums[has_valid] / valid_counts[has_valid]
        
        # 写入新波段
        new_band.WriteArray(new_data)
        new_band.FlushCache()
        
        processed_pixels = np.sum(has_valid)
        return band_idx, processed_pixels, new_height * new_width
    
    # 使用多线程处理波段（但GDAL不是线程安全的，所以顺序处理）
    print("正在处理波段...")
    for band_idx in range(1, band_count + 1):
        band_num, processed_pixels, total_pixels = process_band_optimized(band_idx)
        print(f"波段 {band_num} 处理完成，有效像素: {processed_pixels}/{total_pixels}")
    
    print(f"下采样完成！新数据集尺寸: {new_width}x{new_height}, 分辨率: {target_resolution}米")
    
    return downsampled_dataset

def transform_bbox(origin, target, bbox):
    """
    将包围盒从EPSG:2326转换到EPSG:4326坐标系
    使用GDAL/OSR进行高性能坐标转换
    
    参数:
        bbox: 字典，包含 'min_x', 'max_x', 'min_y', 'max_y'
    
    返回:
        transformed_bbox: 转换后的包围盒字典
    """
    # 创建源坐标系 (EPSG:2326)
    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(origin)
    
    # 创建目标坐标系 (EPSG:4326)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target)
    
    # 创建坐标转换对象
    transform = osr.CoordinateTransformation(source_srs, target_srs)
    
    # 提取包围盒的四个角点
    min_x, max_x = bbox['min_x'], bbox['max_x']
    min_y, max_y = bbox['min_y'], bbox['max_y']
    
    # 定义四个角点坐标 (左下, 右下, 右上, 左上)
    corner_points = np.array([
        [min_x, min_y],  # 左下角
        [max_x, min_y],  # 右下角
        [max_x, max_y],  # 右上角
        [min_x, max_y]   # 左上角
    ], dtype=np.float64)
    
    # 批量转换所有角点坐标
    transformed_points = []
    for point in corner_points:
        # GDAL坐标转换：transform.TransformPoint(x, y)
        transformed_point = transform.TransformPoint(point[0], point[1])
        transformed_points.append([transformed_point[0], transformed_point[1]])
    
    # 转换为numpy数组以便进行向量化计算
    transformed_points = np.array(transformed_points)
    
    # 计算转换后的包围盒范围
    transformed_bbox = {
        'min_x': np.min(transformed_points[:, 0]),
        'max_x': np.max(transformed_points[:, 0]),
        'min_y': np.min(transformed_points[:, 1]),
        'max_y': np.max(transformed_points[:, 1])
    }
    
    return transformed_bbox

def get_dataset_bounds(dataset):
    """

    参数:
        dataset: 一个 GDAL 数据集对象

    返回:
        字典格式：
        {
            "upper_left": {"lon": ..., "lat": ...},
            "lower_left": {"lon": ..., "lat": ...},
            "lower_right": {"lon": ..., "lat": ...},
            "upper_right": {"lon": ..., "lat": ...}
        }
    """
    # 获取地理变换参数
    gt = dataset.GetGeoTransform()
    if gt is None:
        raise ValueError("无地理变换信息")

    # 数据集大小
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 四个角点的原始坐标
    ul_x, ul_y = gt[0], gt[3]  # 左上角
    lr_x, lr_y = gt[0] + width * gt[1] + height * gt[2], gt[3] + width * gt[4] + height * gt[5]  # 右下角
    ll_x, ll_y = ul_x, lr_y  # 左下角
    ur_x, ur_y = lr_x, ul_y  # 右上角

    # 返回结果
    return {
        
        "lower_left": {"lon": ll_x, "lat": ll_y},
        "lower_right": {"lon": lr_x, "lat": lr_y},
        "upper_right": {"lon": ur_x, "lat": ur_y},
        "upper_left": {"lon": ul_x, "lat": ul_y},
    }

def save_info_to_json(info_dict, bounds_dict, huv_stats, output_path):
    """
    将信息字典、边界字典和HUV统计信息合并后保存为 output_path/data.json 文件

    参数:
        info_dict: reproject_dataset 返回的元信息字典（含尺寸信息）
        bounds_dict: get_dataset_bounds_in_4326_dict 返回的边界字典（经纬度）
        huv_stats: process_huv_to_image_from_datasets 返回的HUV统计信息
        output_path: 目录路径，例如 'output/'，将在其中生成 'data.json'
    """
    # 合并为一个 JSON 对象
    combined = {
        "dimensions": {
            "width": info_dict.get("width"),
            "height": info_dict.get("height")
        },
        "huv_stats": huv_stats,
        "bounds_4326": bounds_dict
    }

    # 创建目录（如不存在）
    os.makedirs(output_path, exist_ok=True)

    # 拼接成完整文件路径
    json_path = os.path.join(output_path, 'data.json')

    # 写入 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=4, ensure_ascii=False)

    print(f"信息已保存为 JSON 文件: {json_path}")

def dataset_information_get(dst_dataset):
    if dst_dataset is None:
        raise RuntimeError("源数据无效！")
    # 构建返回信息
    info = {
        "width": dst_dataset.RasterXSize,
        "height": dst_dataset.RasterYSize,
        "bands": []
    }

    for band_idx in range(1, dst_dataset.RasterCount + 1):
        band = dst_dataset.GetRasterBand(band_idx)
        stats = band.GetStatistics(True, True)
        
        # 获取波段描述，如果没有则使用默认名称
        band_description = band.GetDescription()
        if not band_description:
            # 如果没有描述，使用默认的波段名称
            default_names = ["U_velocity", "V_velocity", "H_depth"]
            if band_idx <= len(default_names):
                band_description = default_names[band_idx - 1]
            else:
                band_description = f"Band_{band_idx}"
        
        band_info = {
            "band_index": band_idx,
            "band_name": band_description,
            "min": stats[0],
            "max": stats[1]
        }
        info["bands"].append(band_info)
    return info

def reproject_dataset(src_dataset, target_epsg=3857, resampling=gdal.GRA_Bilinear):
    """
    将栅格数据集重新投影到目标坐标系（默认 EPSG:3857），并返回元信息字典
    """
    print("开始投影转换...")

    # 读取原始坐标系
    src_proj = osr.SpatialReference()
    src_wkt = src_dataset.GetProjection()
    if not src_wkt:
        raise RuntimeError("源数据缺少投影信息，请检查输入 DEM 文件。")

    src_proj.ImportFromWkt(src_wkt)

    # 创建目标坐标系
    tgt_proj = osr.SpatialReference()
    tgt_proj.ImportFromEPSG(target_epsg)
    dst_wkt = tgt_proj.ExportToWkt()

    # 获取原始数据的 geoTransform 和尺寸信息
    geo_transform = src_dataset.GetGeoTransform()
    print(geo_transform)
    width = src_dataset.RasterXSize
    height = src_dataset.RasterYSize

    if width <= 0 or height <= 0:
        raise RuntimeError("源数据尺寸异常（宽或高 <= 0）")

    if geo_transform is None:
        raise RuntimeError("源数据缺少地理转换信息（GeoTransform）")

    # 设置 Warp 重采样选项
    warp_options = gdal.WarpOptions(
        dstSRS=dst_wkt,
        resampleAlg=resampling,
        format='MEM',
        multithread=True
    )

    # 尝试执行重投影
    dst_dataset = gdal.Warp('', src_dataset, options=warp_options)

    if dst_dataset is None:
        raise RuntimeError("重投影失败，请检查输入数据或坐标系设置。")

    print("投影转换完成！开始收集信息...")

    return dst_dataset

def save_gdal_dataset_to_tif(dataset, output_path):
    """
    保存 GDAL 数据集为 GeoTIFF 文件。

    参数:
        dataset: GDAL 数据集对象
        output_path: 输出文件路径，例如 "./output.tif"
    """
    print(f"开始保存数据集到文件: {output_path}")

    # 获取 GDAL 驱动
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("无法获取 GDAL GTiff 驱动")

    # 创建输出文件
    output_file = os.path.join(output_path, "huv_M.tif")
    output_dataset = driver.CreateCopy(output_file, dataset, 0)
    if output_dataset is None:
        raise RuntimeError(f"保存文件失败: {output_path}")

    # 刷新缓存并释放资源
    output_dataset.FlushCache()
    output_dataset = None

    print(f"数据集已成功保存到: {output_path}")

# 示例使用
def huv_generator(result_file, output_path, grid_result, bbox):
    
    merged_result, merged_header = merge_with_result_data(grid_result, result_file)

    dataset, pixel_size = create_tiled_datasets(merged_result, bbox=bbox)
    dataset_M = reproject_dataset(dataset)
    save_gdal_dataset_to_tif(dataset_M, output_path)

    # 对数据集进行下采样
    downsampled_dataset0 = downsample_dataset(dataset, pixel_size, target_resolution=20, no_data_value=-9999)
    # 处理下采样后的数据集
    downsampled_dataset = reproject_dataset(downsampled_dataset0)
    info = dataset_information_get(downsampled_dataset)
    bound = get_dataset_bounds(downsampled_dataset)
    
    # 获取HUV统计信息
    huv_stats = process_huv_to_image_from_datasets(downsampled_dataset, output_path)
    
    save_info_to_json(info, bound, huv_stats, output_path)
    if downsampled_dataset is not None:
        downsampled_dataset.FlushCache()
        downsampled_dataset = None
    if downsampled_dataset0 is not None:
        downsampled_dataset0.FlushCache()
        downsampled_dataset0 = None
    if dataset is not None:
        dataset.FlushCache()
        dataset = None
    if dataset_M is not None:
        dataset_M.FlushCache()
        dataset_M = None
        
    with open(f"{output_path}/render.done", 'w', encoding='utf-8', newline='') as f:
        f.write('done')

    return 0


# 示例使用
# if __name__ == "__main__":
#     start_time = time.time()
#     ne_file = "./flow/ne.txt"
#     ns_file = "./flow/ns.txt"
#     result_file = "./flow/result.dat"
#     output_path = "./flow"

#     # 执行解算
#     grid_time = time.time()
#     grid_result, header, bbox = solve_irregular_grid(ne_file, ns_file)
#     print(f"格网创建和填充耗时: {time.time() - grid_time:.2f} 秒")


#     # 合并结果数据
#     create_time = time.time()
#     merged_result, merged_header = merge_with_result_data(grid_result, result_file)
#     print(f"数据集创建和填充耗时: {time.time() - create_time:.2f} 秒")
#     print(merged_header)
#     print(f"格网包围盒: {bbox}")
    
    
#     # 创建并填充GDAL数据集
#     GDAL_time = time.time()
#     dataset, pixel_size = create_tiled_datasets(merged_result, bbox=bbox)
#     print("填充完成！")
    
#     # 对数据集进行下采样
#     downsampled_dataset = downsample_dataset(dataset, pixel_size, target_resolution=20, no_data_value=-9999)
#     print("下采样完成！")
#     if dataset is not None:
#         dataset.FlushCache()
#         dataset = None
#     # 处理下采样后的数据集

#     transformed_bbox = transform_bbox(2326, 4326, bbox)
#     dst_dataset,info = reproject_dataset(downsampled_dataset)
#     bound = get_dataset_bounds_in_4326_dict(dst_dataset)
#     save_info_to_json(info, bound, output_path)
#     if downsampled_dataset is not None:
#         downsampled_dataset.FlushCache()
#         downsampled_dataset = None

#     process_huv_to_image_from_datasets(dst_dataset, output_path)
#     #  {'depth': {'min_value': 0.0, 'max_value': 12.16477}, 'u': {'min_value': -0.37638304, 'max_value': 4.150747}, 'v': {'min_value': -1.333669, 'max_value': 3.714517}}
#     print(f"生成耗时: {time.time() - GDAL_time:.2f} 秒")
#     print(f"总耗时: {time.time() - start_time:.2f} 秒")
#     input()
    # 清理内存
    
    
    
    
    # np.savetxt("./flow/grid_result.txt", merged_result, fmt="%.6f", 
    #            header="id x y length u v h")


    # & conda run --live-stream --name test python e:/uvWorker/GHuvgenerater.py

