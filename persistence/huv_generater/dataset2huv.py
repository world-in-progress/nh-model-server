import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def process_terrain_data_numpy(dem: np.ndarray, min_height: float, max_height: float, use_8bit: bool = True):
    """NumPy向量化的地形数据处理"""
    # 处理NaN值
    nan_mask = np.isnan(dem)
    
    if use_8bit:
        # 8位处理：0-10 映射到 0-255
        result = np.zeros_like(dem, dtype=np.uint8)
        valid_data = dem[~nan_mask]
        if len(valid_data) > 0:
            processed = np.clip(valid_data * 25.6, 0, 255)
            result[~nan_mask] = processed.astype(np.uint8)
    else:
        # 16位处理
        result = np.zeros_like(dem, dtype=np.uint16)
        valid_data = dem[~nan_mask]
        if len(valid_data) > 0:
            height_range = max_height - min_height
            if height_range > 1e-10:  # 避免除零
                normalized = (valid_data - min_height) / height_range
            else:
                normalized = np.zeros_like(valid_data)  # 如果范围为零，设置为0
            processed = np.clip(normalized * 65535.0, 0, 65535)
            result[~nan_mask] = processed.astype(np.uint16)
    
    return result

def process_uv_data_numpy(u_data: np.ndarray, v_data: np.ndarray, 
                         max_velocity: float, min_u: float, max_u: float, 
                         min_v: float, max_v: float):
    """NumPy向量化的UV数据处理"""
    # 处理NaN值
    nan_mask = np.isnan(u_data) | np.isnan(v_data)
    
    # 初始化结果数组
    u_result = np.zeros_like(u_data, dtype=np.uint8)
    v_result = np.zeros_like(v_data, dtype=np.uint8)
    
    # 处理有效数据
    if np.any(~nan_mask):
        u_valid = u_data[~nan_mask]
        v_valid = v_data[~nan_mask]
        
        # 计算速度模长并限制
        magnitude = np.sqrt(u_valid**2 + v_valid**2)
        over_limit = magnitude > max_velocity
        
        if np.any(over_limit):
            scale = np.where(over_limit, max_velocity / magnitude, 1.0)
            u_valid = u_valid * scale
            v_valid = v_valid * scale
        
        # 归一化到0-255，处理除零情况
        u_range = max_u - min_u
        v_range = max_v - min_v
        
        # 避免除零错误
        if u_range > 1e-10:  # 使用很小的阈值而不是零
            u_norm = (u_valid - min_u) / u_range
        else:
            u_norm = np.zeros_like(u_valid)  # 如果范围为零，设置为0
            
        if v_range > 1e-10:
            v_norm = (v_valid - min_v) / v_range
        else:
            v_norm = np.zeros_like(v_valid)
        
        u_processed = np.clip(u_norm * 256.0, 0, 255).astype(np.uint8)
        v_processed = np.clip(v_norm * 256.0, 0, 255).astype(np.uint8)
        
        u_result[~nan_mask] = u_processed
        v_result[~nan_mask] = v_processed
    
    return u_result, v_result

def create_rgba_image_numpy(dem: np.ndarray, u: np.ndarray, v: np.ndarray, dem_is_8bit: bool = True):
    """NumPy向量化的RGBA图像创建"""
    height, width = dem.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    
    if dem_is_8bit:
        # 8位DEM数据
        rgba[:, :, 0] = dem  # Red
        rgba[:, :, 1] = 0    # Green
    else:
        # 16位DEM数据，拆分为高低8位
        dem_16 = dem.astype(np.uint16)
        rgba[:, :, 0] = (dem_16 >> 8) & 0xFF  # Red - 高8位
        rgba[:, :, 1] = dem_16 & 0xFF         # Green - 低8位
    
    rgba[:, :, 2] = u  # Blue - U分量
    rgba[:, :, 3] = v  # Alpha - V分量
    
    return rgba

def create_dem_rgba_image(dem: np.ndarray) -> Image:
    """
    Create an RGBA image from terrain data.
    """
    height, width = dem.shape
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
    
    # 检查数据类型和范围
    if dem.dtype == np.uint8:
        # 8位数据，直接使用
        rgba_image[:, :, 0] = dem  # Red - 存储主要数值
        rgba_image[:, :, 1] = 0    # Green - 保留为0
    else:
        # 16位数据，拆分为高低8位
        rgba_image[:, :, 0] = (dem >> 8) & 0xFF  # Red - 高8位
        rgba_image[:, :, 1] = dem & 0xFF         # Green - 低8位
    
    rgba_image[:, :, 2] = 0    # Blue
    rgba_image[:, :, 3] = 255  # Alpha（不透明）

    return Image.fromarray(rgba_image, mode="RGBA")

def process_terrain_data(dem: np.ndarray):
    """
    Process terrain data - 保持原始数值范围用于正确的PNG编码
    使用NumPy向量化处理
    """
    # 检查数据范围
    min_height, max_height = np.nanmin(dem), np.nanmax(dem)
    print(f"原始数据范围: {min_height:.3f} 到 {max_height:.3f}")
    
    # 如果数据范围在合理范围内（如0-10），直接使用256倍缩放而不是65536
    if max_height <= 10.0:
        # 对于小范围数据，使用8位精度缩放
        result = process_terrain_data_numpy(dem, min_height, max_height, use_8bit=True)
        return result, min_height, max_height
    else:
        # 对于大范围数据，使用原来的16位方法
        result = process_terrain_data_numpy(dem, min_height, max_height, use_8bit=False)
        return result, min_height, max_height

def process_uv_data(u_data: np.ndarray, v_data: np.ndarray, max_velocity: float):
    """
    Process uv data.
    使用NumPy向量化处理
    """
    # 首先计算范围用于归一化
    magnitude = np.sqrt(u_data**2 + v_data**2)
    over_limit = magnitude > max_velocity
    scale_factor = np.where(over_limit, max_velocity / (magnitude + 1e-8), 1.0)
    u_data_scaled = u_data * scale_factor
    v_data_scaled = v_data * scale_factor
    
    # 计算范围
    min_u, max_u = np.nanmin(u_data_scaled), np.nanmax(u_data_scaled)
    min_v, max_v = np.nanmin(v_data_scaled), np.nanmax(v_data_scaled)
    
    # 添加调试信息
    print(f"UV数据范围: U[{min_u:.6f}, {max_u:.6f}], V[{min_v:.6f}, {max_v:.6f}]")
    
    # 检查范围是否过小
    u_range = max_u - min_u
    v_range = max_v - min_v
    if u_range < 1e-10:
        print(f"警告: U数据范围过小 ({u_range:.2e})，可能全为常数")
    if v_range < 1e-10:
        print(f"警告: V数据范围过小 ({v_range:.2e})，可能全为常数")
    
    # 使用NumPy向量化处理
    norm_u_data, norm_v_data = process_uv_data_numpy(
        u_data_scaled, v_data_scaled,
        max_velocity, min_u, max_u, min_v, max_v
    )
    
    return norm_u_data, norm_v_data, min_u, max_u, min_v, max_v

def create_huv_rgba_image(dem: np.ndarray, u: np.ndarray, v: np.ndarray) -> Image:
    """
    Create an RGBA image from terrain and UV data.
    使用NumPy向量化创建图像
    """
    # 检查DEM数据类型
    dem_is_8bit = dem.dtype == np.uint8
    
    # 使用NumPy向量化创建RGBA图像
    rgba_image = create_rgba_image_numpy(dem, u, v, dem_is_8bit)

    return Image.fromarray(rgba_image, mode="RGBA")

def save_image(image: Image, output_path: str) -> None:
    import os
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    image.save(output_path, format="PNG")

def process_huv_to_image_from_datasets_optimized(dataset, output_path: str, file_suffix: str = "") -> None:
    """
    优化的函数，使用NumPy向量化和多线程处理HUV数据从内存数据集到单个图像
    
    参数:
        dataset: 包含3个波段的GDAL数据集 (波段1: U, 波段2: V, 波段3: H)
        output_path: 输出路径
        file_suffix: 文件后缀
    """
    print(f"开始NumPy向量化处理数据集...")
    import time
    start_time = time.time()
    
    def read_band_data(band_idx):
        """多线程读取波段数据"""
        band_data = dataset.GetRasterBand(band_idx).ReadAsArray().astype(np.float32)
        # 将-9999无数据值替换为0，避免处理错误
        band_data[band_data == -9999] = 0.0
        return band_data
    
    # 使用多线程并行读取波段数据
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_u = executor.submit(read_band_data, 1)  # U波段
        future_v = executor.submit(read_band_data, 2)  # V波段
        future_h = executor.submit(read_band_data, 3)  # H波段
        
        u_data = future_u.result()
        v_data = future_v.result()
        height_data = future_h.result()

    print("波段数据统计:")
    # 并行统计数据
    def compute_stats(data, name):
        """计算数据统计信息"""
        valid = data[data != 0]
        if len(valid) > 0:
            return f"{name}: 最小值={np.min(valid):.6f}, 最大值={np.max(valid):.6f}, 平均值={np.mean(valid):.6f}"
        else:
            return f"{name}: 无有效数据"
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_u_stats = executor.submit(compute_stats, u_data, "U速度")
        future_v_stats = executor.submit(compute_stats, v_data, "V速度")
        future_h_stats = executor.submit(compute_stats, height_data, "H深度")
        
        print(future_u_stats.result())
        print(future_v_stats.result())
        print(future_h_stats.result())

    print(f"数据形状: {u_data.shape}")

    # 并行处理数据
    def process_height_task():
        return process_terrain_data(height_data)
    
    def process_uv_task():
        return process_uv_data(u_data, v_data, max_velocity=5.0)
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_height = executor.submit(process_height_task)
        future_uv = executor.submit(process_uv_task)
        
        norm_height_data, _, _ = future_height.result()
        norm_u_data, norm_v_data, _, _, _, _ = future_uv.result()

    # 创建HUV图像
    image = create_huv_rgba_image(norm_height_data, norm_u_data, norm_v_data)

    # 保存图像，使用file_suffix作为文件名后缀
    import os
    if file_suffix:
        output_huv_path = os.path.join(output_path, f'huv_{file_suffix}.png')
    else:
        output_huv_path = os.path.join(output_path, 'huv.png')
    save_image(image, output_huv_path)
    
    print(f"NumPy向量化处理完成，耗时: {time.time() - start_time:.2f} 秒")

def process_huv_to_image_from_datasets(dataset, output_path: str, file_suffix: str = "") -> None:
    """
    保持向后兼容的函数，调用优化版本（使用NumPy向量化和多线程）
    
    参数:
        dataset: 包含3个波段的GDAL数据集 (波段1: U, 波段2: V, 波段3: H)
        output_path: 输出路径
        file_suffix: 文件后缀
    """
    process_huv_to_image_from_datasets_optimized(dataset, output_path, file_suffix)