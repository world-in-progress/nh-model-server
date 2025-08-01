from osgeo import gdal, osr
import numpy as np
from scipy.spatial import Delaunay, KDTree
import os
import taichi as ti
from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging

from .png2huv import process_huv_to_image_from_datasets
import time

logger = logging.getLogger(__name__)

# 设置GDAL日志级别，减少控制台输出
gdal.SetConfigOption('CPL_LOG_ERRORS', 'OFF')
gdal.SetConfigOption('CPL_LOG', 'OFF')

ti.init(arch=ti.gpu)

class GDALTransformer:
    """使用 GDAL 实现的坐标转换器，替代 pyproj.Transformer"""
    
    def __init__(self, source_epsg, target_epsg):
        """
        初始化 GDAL 坐标转换器
        
        Args:
            source_epsg: 源坐标系 EPSG 代码
            target_epsg: 目标坐标系 EPSG 代码
        """
        self.source_epsg = source_epsg
        self.target_epsg = target_epsg
        
        # 创建源坐标系
        self.source_srs = osr.SpatialReference()
        self.source_srs.ImportFromEPSG(source_epsg)
        # 设置传统GIS轴顺序（x=经度, y=纬度）
        self.source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        
        # 创建目标坐标系
        self.target_srs = osr.SpatialReference()
        self.target_srs.ImportFromEPSG(target_epsg)
        # 设置传统GIS轴顺序（x=经度, y=纬度）
        self.target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        
        # 创建坐标转换对象
        self.coord_transform = osr.CoordinateTransformation(self.source_srs, self.target_srs)
    
    @classmethod
    def from_crs(cls, source_crs, target_crs, always_xy=True):
        """
        从 CRS 字符串创建转换器，模拟 pyproj.Transformer.from_crs
        
        Args:
            source_crs: 源坐标系字符串，格式如 "EPSG:2326"
            target_crs: 目标坐标系字符串，格式如 "EPSG:4326" 
            always_xy: 是否始终按 x,y 顺序（GDAL 默认按 x,y，此参数为兼容性）
        
        Returns:
            GDALTransformer 实例
        """
        # 从字符串中提取 EPSG 代码
        source_epsg = int(source_crs.split(':')[1])
        target_epsg = int(target_crs.split(':')[1])
        
        return cls(source_epsg, target_epsg)
    
    def transform(self, x, y):
        """
        执行坐标转换
        
        Args:
            x: x 坐标，可以是单个值或数组
            y: y 坐标，可以是单个值或数组
        
        Returns:
            tuple: (transformed_x, transformed_y)
        """
        # 处理单个点
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            point = self.coord_transform.TransformPoint(x, y)
            return point[0], point[1]
        
        # 处理数组
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        
        # 确保是1维数组
        x_flat = x_array.flatten()
        y_flat = y_array.flatten()
        
        # 批量转换
        transformed_x = []
        transformed_y = []
        
        for xi, yi in zip(x_flat, y_flat):
            point = self.coord_transform.TransformPoint(xi, yi)
            transformed_x.append(point[0])
            transformed_y.append(point[1])
        
        # 转换为 numpy 数组并恢复原始形状
        transformed_x = np.array(transformed_x).reshape(x_array.shape)
        transformed_y = np.array(transformed_y).reshape(y_array.shape)
        
        return transformed_x, transformed_y

@ti.kernel
def interpolate_taichi_multi_field(
    px: ti.types.ndarray(), py: ti.types.ndarray(),  # type: ignore
    ax: ti.types.ndarray(), ay: ti.types.ndarray(), # type: ignore
    bx: ti.types.ndarray(), by_: ti.types.ndarray(), # type: ignore
    cx: ti.types.ndarray(), cy: ti.types.ndarray(), # type: ignore
    z0_fields: ti.types.ndarray(), z1_fields: ti.types.ndarray(), z2_fields: ti.types.ndarray(), # type: ignore
    valid: ti.types.ndarray(), results: ti.types.ndarray(), # type: ignore
    min_values: ti.types.ndarray(), max_values: ti.types.ndarray() # type: ignore
):
    for i in range(px.shape[0]):
        if valid[i] == 1:
            x = px[i]; y = py[i]
            x0 = ax[i]; y0 = ay[i]
            x1 = bx[i]; y1 = by_[i]
            x2 = cx[i]; y2 = cy[i]

            det = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
            l0 = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) / det
            l1 = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) / det
            l2 = 1.0 - l0 - l1

            # 处理多个字段
            for field_idx in range(z0_fields.shape[1]):
                interpolated_value = l0 * z0_fields[i, field_idx] + l1 * z1_fields[i, field_idx] + l2 * z2_fields[i, field_idx]
                # 应用 clipping
                if interpolated_value < min_values[field_idx]:
                    interpolated_value = min_values[field_idx]
                elif interpolated_value > max_values[field_idx]:
                    interpolated_value = max_values[field_idx]
                results[i, field_idx] = interpolated_value
        else:
            for field_idx in range(z0_fields.shape[1]):
                results[i, field_idx] = 1e-10

class TINContext:
    def __init__(self):
        self.tin = None
        self.extended_points = None
        self.grid_points = None
        self.num_points = None
        self.vertices = None
        self.extended_points_hash = None

    def reset(self):
        self.__init__()

    def update_extended_points(self, points):
        self.extended_points = points
        self.extended_points_hash = hashlib.sha256(points.tobytes()).hexdigest()


def calculate_boundary_points(points, boundary_buffer, n_boundary_points=10):
    """计算扩展包围盒和边界点"""
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_buffer = x_range * boundary_buffer
    y_buffer = y_range * boundary_buffer

    ext_x_min = x_min - x_buffer
    ext_x_max = x_max + x_buffer
    ext_y_min = y_min - y_buffer
    ext_y_max = y_max + y_buffer

    boundary_points = []
    for i in range(n_boundary_points):
        x = ext_x_min + (ext_x_max - ext_x_min) * i / (n_boundary_points - 1)
        boundary_points.append([x, ext_y_min])
    for i in range(n_boundary_points):
        y = ext_y_min + (ext_y_max - ext_y_min) * i / (n_boundary_points - 1)
        boundary_points.append([ext_x_max, y])
    for i in range(n_boundary_points):
        x = ext_x_max - (ext_x_max - ext_x_min) * i / (n_boundary_points - 1)
        boundary_points.append([x, ext_y_max])
    for i in range(n_boundary_points):
        y = ext_y_max - (ext_y_max - ext_y_min) * i / (n_boundary_points - 1)
        boundary_points.append([ext_x_min, y])

    return np.array(boundary_points)

def merge_points(points, boundary_points):
    """合并原始点和边界点"""
    return np.vstack([points, boundary_points])

def construct_delaunay(extended_points):
    """构建三角网"""
    return Delaunay(extended_points)

def prepare_grid_data_optimized(tri, xi, yi, batch_size=50000):
    """
    优化的网格数据准备函数 - 使用向量化操作和智能批处理
    """
    # 向量化展平和组合网格点
    grid_points = np.column_stack([xi.ravel(), yi.ravel()])
    num_points = len(grid_points)
    
    # 直接使用三角剖分的 find_simplex 方法（更准确）
    logger.debug(f"正在处理 {num_points:,} 个网格点...")
    
    if num_points <= batch_size:
        # 小数据集直接处理
        simplex_ids = tri.find_simplex(grid_points)
    else:
        # 大数据集分批处理
        simplex_ids = np.empty(num_points, dtype=np.int32)
        n_batches = (num_points + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)
            batch_points = grid_points[start_idx:end_idx]
            simplex_ids[start_idx:end_idx] = tri.find_simplex(batch_points)
    
    # 筛选有效网格点
    valid_mask = simplex_ids >= 0
    valid_simplex_ids = simplex_ids[valid_mask]
    vertices = tri.simplices[valid_simplex_ids]
    
    # 修复溢出问题：使用浮点数计算
    valid_count = int(np.sum(valid_mask))
    valid_percentage = 100.0 * valid_count / num_points
    logger.debug(f"有效网格点: {valid_count:,}/{num_points:,} ({valid_percentage:.1f}%)")
    
    return grid_points, num_points, vertices

def build_triangulation(context: TINContext, points, xi=None, yi=None, add_boundary=True, boundary_buffer=0.1, use_global_tin=True):
    """
    构建三角网，支持全局三角网逻辑，处理三角网、扩展点集和网格点准备
    """
    if add_boundary:
        boundary_points = calculate_boundary_points(points, boundary_buffer)
        extended_points = merge_points(points, boundary_points)
    else:
        extended_points = points

    extended_points_hash = hashlib.sha256(extended_points.tobytes()).hexdigest()

    if use_global_tin and context.tin is not None and context.extended_points_hash == extended_points_hash:
        logger.debug("使用已存在的全局三角网，跳过三角网构建...")
        return context.tin, extended_points, context.grid_points, context.num_points, context.vertices

    logger.debug("构建新的三角网...")
    tri = construct_delaunay(extended_points)
    
    # 如果提供了网格坐标，则准备网格点数据
    if xi is not None and yi is not None:
        logger.debug("准备网格点数据...")
        prep_start = time.time()
        grid_points, num_points, vertices = prepare_grid_data_optimized(tri, xi, yi)

        if use_global_tin:
            context.grid_points = grid_points
            context.num_points = num_points
            context.vertices = vertices
            logger.debug("网格点数据已保存为上下文变量")
        
        logger.debug(f"网格点数据准备耗时: {time.time() - prep_start:.2f} 秒")
    
    if use_global_tin:
        context.tin = tri
        context.update_extended_points(extended_points)
        logger.debug("三角网已保存为上下文变量")

    return tri, extended_points, grid_points, num_points, vertices

def reset_global_tin():
    """重置全局三角网变量"""
    global GLOBAL_TIN_CONTEXT
    GLOBAL_TIN_CONTEXT.reset()
    logger.debug("全局三角网已重置")

def prepare_field_data_parallel(data_array, header, fields, context):
    """并行准备字段数据 - 优化版本，适配numpy数组输入"""
    
    def prepare_single_field(field):
        """优化的单字段数据准备"""
        try:
            field_start_time = time.time()
            logger.debug(f"准备字段数据: {field}")

            # 从numpy数组中提取字段值
            field_idx = header.index(field)
            field_values = data_array[:, field_idx].astype(np.float32)
            n_data_points = len(field_values)
            
            # 构建 KDTree（仅对原始数据点）
            kdtree = KDTree(context.extended_points[:n_data_points])
            
            # 向量化估计边界点值
            boundary_points = context.extended_points[n_data_points:]
            n_boundary = len(boundary_points)
            
            if n_boundary > 0:
                # 批量查询最近邻（k=3以提高稳定性）
                distances, indices = kdtree.query(boundary_points, k=min(5, n_data_points))
                
                # 处理距离为0的情况（避免除零错误）
                zero_dist_mask = distances[:, 0] == 0
                boundary_values = np.zeros(n_boundary, dtype=np.float32)
                
                # 距离为0时直接使用最近点的值
                boundary_values[zero_dist_mask] = field_values[indices[zero_dist_mask, 0]]
                
                # 距离不为0时使用逆距离加权
                non_zero_mask = ~zero_dist_mask
                if np.any(non_zero_mask):
                    weights = 1.0 / distances[non_zero_mask]
                    weights = weights / weights.sum(axis=1, keepdims=True)
                    boundary_values[non_zero_mask] = np.sum(
                        field_values[indices[non_zero_mask]] * weights, axis=1
                    )
                
                extended_values = np.concatenate([field_values, boundary_values])
            else:
                extended_values = field_values
            
            logger.debug(f"字段 {field} 数据准备耗时: {time.time() - field_start_time:.2f} 秒")
            return field, extended_values, field_values

        except Exception as e:
            logger.error(f"准备字段 '{field}' 数据时出错: {e}")
            return field, None, None

    # 并行处理所有字段
    with ThreadPoolExecutor(max_workers=min(len(fields), 4)) as executor:
        futures = [executor.submit(prepare_single_field, field) for field in fields]
        field_data = {future.result()[0]: future.result()[1:] for future in futures}

    return field_data

def perform_taichi_computation(context, field_data, xi, yi):
    """批量进行 Taichi GPU 计算 - 优化版本"""
    logger.debug("开始批量 Taichi GPU 计算...")
    taichi_start = time.time()

    # 提取几何数据
    grid_points, num_points = context.grid_points, context.num_points
    simplex_ids = context.tin.find_simplex(grid_points)
    valid_mask = simplex_ids >= 0
    vertices = context.tin.simplices[simplex_ids[valid_mask]]

    # 提取网格点坐标
    px, py = grid_points[:, 0], grid_points[:, 1]

    # 批量提取三角形顶点坐标
    triangle_vertices = context.extended_points[vertices]  # shape: (n_valid_triangles, 3, 2)
    v0, v1, v2 = triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2]

    # 筛选有效字段数据
    valid_field_items = [(field, data) for field, data in field_data.items() 
                        if data[0] is not None]
    
    if not valid_field_items:
        logger.warning("没有有效的字段数据")
        return {field: None for field in field_data.keys()}

    # 解包字段数据
    valid_fields, field_values = zip(*valid_field_items)
    all_extended_values, all_original_values = zip(*field_values)

    # 向量化组织字段数据矩阵
    extended_array = np.array(all_extended_values)  # shape: (n_fields, n_extended_points)
    original_array = np.array(all_original_values)  # shape: (n_fields, n_original_points)
    
    # 批量提取顶点字段值
    z0_fields = extended_array[:, vertices[:, 0]].T  # shape: (n_valid_triangles, n_fields)
    z1_fields = extended_array[:, vertices[:, 1]].T
    z2_fields = extended_array[:, vertices[:, 2]].T

    # 向量化计算最值
    min_values = np.nanmin(original_array, axis=1).astype(np.float32)
    max_values = np.nanmax(original_array, axis=1).astype(np.float32)

    # 初始化结果数组
    results = np.full((num_points, len(valid_fields)), np.nan, dtype=np.float32)

    # 调用 Taichi kernel（确保数据连续性）
    interpolate_taichi_multi_field(
        np.ascontiguousarray(px, dtype=np.float32),
        np.ascontiguousarray(py, dtype=np.float32),
        np.ascontiguousarray(v0[:, 0], dtype=np.float32),
        np.ascontiguousarray(v0[:, 1], dtype=np.float32),
        np.ascontiguousarray(v1[:, 0], dtype=np.float32),
        np.ascontiguousarray(v1[:, 1], dtype=np.float32),
        np.ascontiguousarray(v2[:, 0], dtype=np.float32),
        np.ascontiguousarray(v2[:, 1], dtype=np.float32),
        np.ascontiguousarray(z0_fields, dtype=np.float32),
        np.ascontiguousarray(z1_fields, dtype=np.float32),
        np.ascontiguousarray(z2_fields, dtype=np.float32),
        np.ascontiguousarray(valid_mask, dtype=np.int32),
        results,
        min_values, max_values
    )

    # 重组结果为字典格式
    field_arrays = {
        field: (results[:, idx].reshape(xi.shape) 
                if field in valid_fields 
                else None)
        for idx, field in enumerate(list(valid_fields) + 
                                  [f for f in field_data.keys() if f not in valid_fields])
    }
    
    # 处理无效点
    for field in valid_fields:
        if field_arrays[field] is not None:
            field_arrays[field][~valid_mask.reshape(xi.shape)] = np.nan

    logger.debug(f"批量 Taichi GPU 计算耗时: {time.time() - taichi_start:.2f} 秒")
    return field_arrays

def create_datasets_parallel(field_arrays, cols, rows, xmin, ymin, ymax, pixel_size_deg, epsg_code):
    """并行创建数据集"""
    def create_dataset(field):
        try:
            field_start_time = time.time()
            logger.debug(f"创建数据集: {field}")

            field_array = field_arrays[field]
            if field_array is None:
                return field, None, {'error': '字段数据准备失败'}

            # 由于我们的Y坐标网格是从下到上递增的，需要翻转数组以符合栅格图像格式
            # 栅格图像的第一行应该对应最大的Y坐标值
            field_array_flipped = np.flipud(field_array)

            # 创建内存dataset
            driver = gdal.GetDriverByName('MEM')
            dataset = driver.Create('', cols, rows, 1, gdal.GDT_Float32)

            if dataset is None:
                raise RuntimeError(f"无法创建内存栅格数据集")

            # 设置地理变换参数 - 使用标准的栅格坐标系
            # 第一行对应ymax，最后一行对应ymin
            geotransform = (xmin, pixel_size_deg, 0, ymax, 0, -pixel_size_deg)
            dataset.SetGeoTransform(geotransform)

            # 设置空间参考系统
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg_code)
            dataset.SetProjection(srs.ExportToWkt())

            # 写入数据到第一个波段
            band = dataset.GetRasterBand(1)
            band.WriteArray(field_array_flipped)  # 使用翻转后的数组
            band.SetNoDataValue(np.nan)
            band.SetDescription(field)
            band.FlushCache()

            # 记录结果统计（使用原始数组计算统计信息）
            field_result = {
                'min_value': np.nanmin(field_array),
                'max_value': np.nanmax(field_array),
                # 'mean_value': np.nanmean(field_array)
            }

            logger.debug(f"  {field}字段统计: min={field_result['min_value']:.4f}, "
                        f"max={field_result['max_value']:.4f}")
                        # f"mean={field_result['mean_value']:.4f}")

            logger.debug(f"数据集 {field} 创建耗时: {time.time() - field_start_time:.2f} 秒")

            return field, dataset, field_result

        except Exception as e:
            logger.error(f"创建字段 '{field}' 数据集时出错: {e}")
            return field, None, {'error': str(e)}

    with ThreadPoolExecutor(max_workers=len(field_arrays)) as executor:
        futures = [executor.submit(create_dataset, field) for field in field_arrays.keys()]
        field_results = {future.result()[0]: future.result()[1:] for future in futures}

    return field_results

def generate_raster_grid(data_array, header, pixel_size, epsg_code):
    """
    根据输入数据生成栅格化的网格坐标参数。
    
    参数:
        data_array: numpy数组，形状为(n_rows, n_columns)
        header: 字段名列表
        pixel_size: 像素大小
        epsg_code: 坐标系代码
    """
    # 找到x和y字段的索引
    x_idx = header.index('x')
    y_idx = header.index('y')
    
    # 提取坐标
    x_coords = data_array[:, x_idx]
    y_coords = data_array[:, y_idx]

    # 确定散点数据的包围盒边界范围
    xmin, xmax = x_coords.min(), x_coords.max()
    ymin, ymax = y_coords.min(), y_coords.max()

    # 根据坐标系调整像素大小单位和缓冲区
    if epsg_code == 4326:
        pixel_size_deg = pixel_size / 111000.0  # 将米转换为度
        buffer = pixel_size_deg * 0.5
    else:
        pixel_size_deg = pixel_size
        buffer = pixel_size * 0.5

    # 添加缓冲区
    xmin -= buffer
    xmax += buffer
    ymin -= buffer
    ymax += buffer

    # 计算输出栅格的行列数
    cols = int(np.ceil((xmax - xmin) / pixel_size_deg))
    rows = int(np.ceil((ymax - ymin) / pixel_size_deg))

    logger.debug(f"创建三个独立的内存TIF数据集")
    logger.debug(f"栅格尺寸: {cols} x {rows}")
    logger.debug(f"数据点范围: x=[{x_coords.min():.2f}, {x_coords.max():.2f}], y=[{y_coords.min():.2f}, {y_coords.max():.2f}]")
    logger.debug(f"栅格范围: x=[{xmin:.2f}, {xmax:.2f}], y=[{ymin:.2f}, {ymax:.2f}]")
    logger.debug(f"像素大小: {pixel_size_deg:.6f}")

    # 创建输出栅格的坐标网格
    x_coords_grid = np.linspace(xmin, xmax, cols)
    y_coords_grid = np.linspace(ymin, ymax, rows)  # 修正：从下到上递增，与数据坐标一致
    xi, yi = np.meshgrid(x_coords_grid, y_coords_grid)

    # 组织输入数据点
    points = np.column_stack((x_coords, y_coords))

    return xmin, xmax, ymin, ymax, xi, yi, cols, rows, points, pixel_size_deg

def generate_all_tifs(data_array: np.ndarray, header: list, pixel_size: float = 10.0, 
                     epsg_code: int = 4326, boundary_buffer: float = 0.1,
                     use_global_tin: bool = True, GLOBAL_TIN_CONTEXT: TINContext = None):
    """
    生成三个单独的内存dataset，按顺序为depth, u, v
    
    参数:
        data_array: numpy数组，形状为(n_rows, n_columns)
        header: 字段名列表
        其他参数保持不变
    """
    # 检查必要字段是否存在
    required_fields = ['depth', 'u', 'v']
    for field in required_fields:
        if field not in header:
            raise ValueError(f"字段 '{field}' 在数据中不存在")

    xmin, xmax, ymin, ymax, xi, yi, cols, rows, points, pixel_size_deg = generate_raster_grid(data_array, header, pixel_size, epsg_code)

    # 构建三角网
    start_time = time.time()
    tri, extended_points, grid_points, num_points, vertices = build_triangulation(
        GLOBAL_TIN_CONTEXT,
        points=points,
        xi=xi,
        yi=yi,
        add_boundary=True,
        boundary_buffer=boundary_buffer,
        use_global_tin=use_global_tin
    )
    logger.debug(f"三角网构建耗时: {time.time() - start_time:.2f} 秒")
    logger.debug("----------------------------------------------------------")

    # 并行准备字段数据
    field_data = prepare_field_data_parallel(data_array, header, required_fields, GLOBAL_TIN_CONTEXT)

    # 批量进行 Taichi GPU 计算
    field_arrays = perform_taichi_computation(GLOBAL_TIN_CONTEXT, field_data, xi, yi)

    # 并行创建数据集
    field_results = create_datasets_parallel(field_arrays, cols, rows, xmin, ymin, ymax, pixel_size_deg, epsg_code)

    # 按照指定顺序添加到datasets列表中
    datasets = []
    results = {}
    field_stats = {}  # 存储每个字段的统计信息
    
    for field in required_fields:
        dataset, result = field_results[field]
        datasets.append(dataset)
        results[field] = result
        
        # 提取并存储每个字段的最大最小值
        if result is not None and 'error' not in result:
            field_stats[field] = {
                'min_value': result['min_value'],
                'max_value': result['max_value']
            }
        else:
            field_stats[field] = {
                'min_value': None,
                'max_value': None
            }

    # 图片尺寸信息
    image_size = {
        'width': cols,
        'height': rows
    }

    # 栅格边界信息（用于坐标转换）
    raster_bounds = (xmin, xmax, ymin, ymax)

    logger.debug(f"三个独立的内存TIF数据集创建完成")

    return datasets, field_stats, image_size, raster_bounds  # 返回dataset列表、字段统计信息、图片尺寸和栅格边界


def data_process_alternative(ne_file_path, result_file_path, transform_crs=True, source_epsg=2326, target_epsg=4326):
    """
    从文件中读取点数据并与结果数据合并
    
    Args:
        ne_file_path: 点数据文件路径
        result_file_path: 结果数据文件路径
        transform_crs: 是否进行坐标转换
        source_epsg: 源坐标系EPSG代码
        target_epsg: 目标坐标系EPSG代码
    
    Returns:
        tuple: (data_array, header)
    """
    print("开始读取ne点文件...")
    # 读取点数据文件
    ne_raw = np.genfromtxt(ne_file_path, dtype=np.float64, delimiter=",")
    
    # 从倒数第4列和第3列提取x,y坐标
    xe_list = ne_raw[:, -4]  # 倒数第4列作为x坐标
    ye_list = ne_raw[:, -3]  # 倒数第3列作为y坐标
    
    # 确保两个列表长度相同
    if len(xe_list) != len(ye_list):
        raise ValueError(f"xe_list和ye_list长度不匹配: {len(xe_list)} vs {len(ye_list)}")
    
    print(f"成功读取 {len(xe_list)} 个点")
    print(f"坐标范围: x=[{xe_list.min():.2f}, {xe_list.max():.2f}], y=[{ye_list.min():.2f}, {ye_list.max():.2f}]")
    
    # 坐标转换
    if transform_crs:
        print(f"开始坐标转换: EPSG:{source_epsg} -> EPSG:{target_epsg}")
        transformer = GDALTransformer.from_crs(f"EPSG:{source_epsg}", f"EPSG:{target_epsg}", always_xy=True)
        
        # 批量转换坐标以提高效率
        transformed_x, transformed_y = transformer.transform(xe_list, ye_list)
        xe_list = np.array(transformed_x, dtype=np.float64)
        ye_list = np.array(transformed_y, dtype=np.float64)
        print("坐标转换完成")
        print(f"转换后坐标范围: x=[{xe_list.min():.2f}, {xe_list.max():.2f}], y=[{ye_list.min():.2f}, {ye_list.max():.2f}]")
    
    print("开始读取result数据文件...")
    results_raw = np.genfromtxt(result_file_path, dtype=np.float64, skip_header=1, delimiter="\t")
    
    # 检查数据长度并取较短的长度
    ne_length = len(xe_list)
    result_length = len(results_raw)
    
    print(f"ne文件数据行数: {ne_length}")
    print(f"result文件数据行数: {result_length}")
    
    # 取两个文件中较短的长度，避免长度不匹配问题
    min_length = min(ne_length, result_length)
    
    if ne_length != result_length:
        print(f"警告: 数据长度不匹配，将使用较短的长度: {min_length}")
        
        # 截取数据到相同长度
        xe_list = xe_list[:min_length]
        ye_list = ye_list[:min_length]
        results_raw = results_raw[:min_length]
    
    # 直接按行顺序组合数据，不需要通过ID匹配
    # 假设数据是按相同顺序排列的
    point_ids = np.arange(min_length, dtype=np.float64)
    
    # 提取结果数据
    # result文件格式：grid_id h u v ze
    # 列索引：0=grid_id, 1=h, 2=u, 3=v, 4=ze
    results_data = results_raw[:, [2, 3]]  # u, v (第2列和第3列)
    depth_values = np.maximum(0.0, (results_raw[:, 1]-results_raw[:, 4]))  # h (第1列)
    
    print(f"深度值范围: [{depth_values.min():.3f}, {depth_values.max():.3f}]")
    print(f"U速度范围: [{results_data[:, 0].min():.3f}, {results_data[:, 0].max():.3f}]")
    print(f"V速度范围: [{results_data[:, 1].min():.3f}, {results_data[:, 1].max():.3f}]")
    print(f"非零深度点数: {np.sum(depth_values > 0)}/{len(depth_values)}")
    
    # 直接按行组合数据
    combined_data = np.column_stack([
        point_ids,      # id
        xe_list,        # x
        ye_list,        # y
        results_data,   # u, v
        depth_values    # depth
    ])
    
    print(f"成功处理 {len(combined_data)} 条数据")
    print(f"合并后数据范围检查:")
    print(f"  X坐标: [{combined_data[:, 1].min():.2f}, {combined_data[:, 1].max():.2f}]")
    print(f"  Y坐标: [{combined_data[:, 2].min():.2f}, {combined_data[:, 2].max():.2f}]")
    print(f"  深度: [{combined_data[:, 5].min():.3f}, {combined_data[:, 5].max():.3f}]")
    
    # 创建表头字符串列表
    header = ['id', 'x', 'y', 'u', 'v', 'depth']
    # 返回纯数据数组
    data_array = combined_data
    
    return data_array, header

def huv_generater(ne_list, result_path, GLOBAL_TIN_CONTEXT, output_path, epsg_code = 3857):

    pixel_size = 5.0  # 5米分辨率（EPSG:4326会自动转换为度，EPSG:3857保持米单位）
    boundary_buffer = 0.1  # 边界缓冲区比例

    try:
        # 加载数据并进行坐标转换
        data_array, header = data_process_alternative(ne_list, result_path, transform_crs=True, source_epsg=2326, target_epsg=epsg_code)

        # 对于第一个文件，重置全局三角网
        if GLOBAL_TIN_CONTEXT == None:
            reset_global_tin()
            logger.info("开始处理第一个文件，将构建全局三角网...")

        # 生成三个独立的dataset(depth, u, v)
        datasets, field_stats, image_size, raster_bounds = generate_all_tifs(
            data_array=data_array,
            header=header,
            pixel_size=pixel_size,
            epsg_code=epsg_code,
            boundary_buffer=boundary_buffer,
            use_global_tin=True,   # 启用全局三角网
            GLOBAL_TIN_CONTEXT=GLOBAL_TIN_CONTEXT
        )
        
        # 记录字段统计信息和图片尺寸
        # print(f"字段统计信息: {field_stats['min_value']},{field_stats['min_value']}")
        print(f"图片尺寸: {image_size['width']} x {image_size['height']}")

        # 计算图像在4326坐标系下的边界坐标
        # 使用栅格的实际边界（包含缓冲区）
        xmin_raster, xmax_raster, ymin_raster, ymax_raster = raster_bounds
        
        # 转换边界坐标到4326坐标系
        transformer_to_4326 = GDALTransformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
        
        # 转换左上角和右下角坐标（使用栅格边界）
        top_left_lon, top_left_lat = transformer_to_4326.transform(xmin_raster, ymax_raster)
        bottom_right_lon, bottom_right_lat = transformer_to_4326.transform(xmax_raster, ymin_raster)

        process_huv_to_image_from_datasets(datasets, output_path)
        output_stats_path = output_path + 'huv_stats.txt'
        # 写入统计信息到txt文件
        try:
            with open(output_stats_path, 'w', encoding='utf-8') as stats_file:
                # 写入表头
                stats_file.write("width height waterHeightMin waterHeightMax velocityUMin velocityUMax velocityVMin velocityVMax topLeftLon topLeftLat bottomRightLon bottomRightLat\n")
                # 写入数据
                stats_file.write(f"{image_size['width']} {image_size['height']} {field_stats['depth']['min_value']} {field_stats['depth']['max_value']} {field_stats['u']['min_value']} {field_stats['u']['max_value']} {field_stats['v']['min_value']} {field_stats['v']['max_value']} {top_left_lon} {top_left_lat} {bottom_right_lon} {bottom_right_lat}\n")
            logger.debug(f"统计信息已写入: {output_stats_path}")
        except Exception as e:
            logger.error(f"写入统计信息文件时出错: {e}")

        # 释放dataset资源
        for i, dataset in enumerate(datasets):
            if dataset is not None:
                dataset.FlushCache()
                dataset = None
        
        with open(f"{output_path}render.done", 'w', encoding='utf-8', newline='') as f:
            f.write('done')

    except Exception as e:
        logger.error(f"处理文件时出错: {e}", exc_info=True)

    return 0

def extract_xy_to_dict(data: str) -> dict:
    xe_list = []
    ye_list = []
    
    with open(data, 'r') as file:
        lines = file.readlines()
        # header = lines[0].strip().split(", ")  # 读取表头
        
        # 跳过表头，从第二行开始处理
    for line in lines[1:]:
        # 使用逗号分割并去除空格
        values = [value.strip() for value in line.split(", ")]
        
        # 根据列的顺序：id, x, y, z, level, u, v, depth
        # x 是第2列（索引1），y 是第3列（索引2）
        xe_list.append(float(values[1]))  # 提取x列
        ye_list.append(float(values[2]))  # 提取y列
    
    return {
        'xe_list': xe_list,
        'ye_list': ye_list
    }

if __name__ == "__main__":

    # 全局变量用于存储三角网
    GLOBAL_TIN_CONTEXT = TINContext()

    # ne_dict=extract_xy_to_dict('./huv/huvResult_0.txt')
    # huv_generater(ne_dict, './huv/huvResult_0.txt', GLOBAL_TIN_CONTEXT, './step0/')

    huv_generater('./data/simulation0719/ne.txt', './data/simulation0719/result.dat', GLOBAL_TIN_CONTEXT, './step0/')

    