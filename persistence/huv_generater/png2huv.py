from osgeo import gdal
import numpy as np
from PIL import Image
import taichi as ti

# 初始化 Taichi GPU 加速
ti.init(arch=ti.gpu)

def load_terrain_data(filename: str) -> np.ndarray:
    """
    Load terrain data from a GeoTIFF file.
    """
    dataset = gdal.Open(filename)
    if dataset is None:
        raise ValueError("Failed to open file: {}".format(filename))
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    return data

@ti.kernel
def process_terrain_data_gpu(
    dem: ti.types.ndarray(), # type: ignore
    result: ti.types.ndarray(), # type: ignore
    min_height: ti.f32, max_height: ti.f32, use_8bit: ti.i32 # type: ignore
):
    """GPU加速的地形数据处理"""
    for i, j in ti.ndrange(dem.shape[0], dem.shape[1]):
        if ti.math.isnan(dem[i, j]):
            result[i, j] = 0
        else:
            if use_8bit == 1:
                # 8位处理：0-10 映射到 0-255
                result[i, j] = ti.cast(ti.max(0.0, ti.min(255.0, dem[i, j] * 25.6)), ti.uint8)
            else:
                # 16位处理
                normalized = (dem[i, j] - min_height) / (max_height - min_height)
                result[i, j] = ti.cast(ti.max(0.0, ti.min(65535.0, normalized * 65535.0)), ti.uint16)

@ti.kernel
def process_uv_data_gpu(
    u_data: ti.types.ndarray(), v_data: ti.types.ndarray(), # type: ignore
    u_result: ti.types.ndarray(), v_result: ti.types.ndarray(), # type: ignore
    max_velocity: ti.f32, min_u: ti.f32, max_u: ti.f32, min_v: ti.f32, max_v: ti.f32 # type: ignore
):
    """GPU加速的UV数据处理"""
    for i, j in ti.ndrange(u_data.shape[0], u_data.shape[1]):
        u_val = u_data[i, j]
        v_val = v_data[i, j]
        
        if ti.math.isnan(u_val) or ti.math.isnan(v_val):
            u_result[i, j] = 0
            v_result[i, j] = 0
        else:
            # 计算速度模长并限制
            magnitude = ti.sqrt(u_val * u_val + v_val * v_val)
            if magnitude > max_velocity:
                scale = max_velocity / magnitude
                u_val *= scale
                v_val *= scale
            
            # 归一化到0-255
            u_norm = (u_val - min_u) / (max_u - min_u)
            v_norm = (v_val - min_v) / (max_v - min_v)
            
            u_result[i, j] = ti.cast(ti.max(0.0, ti.min(255.0, u_norm * 256.0)), ti.uint8)
            v_result[i, j] = ti.cast(ti.max(0.0, ti.min(255.0, v_norm * 256.0)), ti.uint8)

@ti.kernel
def create_rgba_image_gpu(
    dem: ti.types.ndarray(), u: ti.types.ndarray(), v: ti.types.ndarray(), # type: ignore
    rgba: ti.types.ndarray(), dem_is_8bit: ti.i32 # type: ignore
):
    """GPU加速的RGBA图像创建"""
    for i, j in ti.ndrange(dem.shape[0], dem.shape[1]):
        if dem_is_8bit == 1:
            # 8位DEM数据
            rgba[i, j, 0] = dem[i, j]  # Red
            rgba[i, j, 1] = 0          # Green
        else:
            # 16位DEM数据，拆分为高低8位
            dem_val = ti.cast(dem[i, j], ti.uint16)
            rgba[i, j, 0] = (dem_val >> 8) & 0xFF  # Red - 高8位
            rgba[i, j, 1] = dem_val & 0xFF         # Green - 低8位
        
        rgba[i, j, 2] = u[i, j]  # Blue - U分量
        rgba[i, j, 3] = v[i, j]  # Alpha - V分量

def process_terrain_data(dem: np.ndarray):
    """
    Process terrain data - 保持原始数值范围用于正确的PNG编码
    使用GPU加速处理
    """
    # 检查数据范围
    min_height, max_height = np.nanmin(dem), np.nanmax(dem)
    print(f"原始数据范围: {min_height:.3f} 到 {max_height:.3f}")
    
    # 如果数据范围在合理范围内（如0-10），直接使用256倍缩放而不是65536
    if max_height <= 10.0:
        # 对于小范围数据，使用8位精度缩放
        result = np.zeros_like(dem, dtype=np.uint8)
        process_terrain_data_gpu(dem, result, min_height, max_height, 1)
        return result, min_height, max_height
    else:
        # 对于大范围数据，使用原来的16位方法
        result = np.zeros_like(dem, dtype=np.uint16)
        process_terrain_data_gpu(dem, result, min_height, max_height, 0)
        return result, min_height, max_height

def process_uv_data(u_data: np.ndarray, v_data: np.ndarray, max_velocity: float):
    """
    Process uv data.
    使用GPU加速处理
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
    
    # 使用GPU处理
    norm_u_data = np.zeros_like(u_data_scaled, dtype=np.uint8)
    norm_v_data = np.zeros_like(v_data_scaled, dtype=np.uint8)
    
    process_uv_data_gpu(
        u_data_scaled, v_data_scaled,
        norm_u_data, norm_v_data,
        max_velocity, min_u, max_u, min_v, max_v
    )
    
    return norm_u_data, norm_v_data, min_u, max_u, min_v, max_v

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

def create_huv_rgba_image(dem: np.ndarray, u: np.ndarray, v: np.ndarray) -> Image:
    """
    Create an RGBA image from terrain and UV data.
    使用GPU加速创建图像
    """
    height, width = dem.shape
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
    
    # 检查DEM数据类型
    dem_is_8bit = 1 if dem.dtype == np.uint8 else 0
    
    # 使用GPU加速创建RGBA图像
    create_rgba_image_gpu(dem, u, v, rgba_image, dem_is_8bit)

    return Image.fromarray(rgba_image, mode="RGBA")

def save_image(image: Image, output_path: str) -> None:
    image.save(output_path, format="PNG")

def process_dem_to_image(input_dem_path: str, output_dem_path: str, file) -> None:
    """
    Process DEM data to an image."
    """
    data = load_terrain_data(input_dem_path)
    norm_data, min_height, max_height = process_terrain_data(data)
    image = create_dem_rgba_image(norm_data)
    save_image(image, output_dem_path)
    file.write(f"DEM\t{min_height}\t{max_height}\n")

def encode_huv_to_image(input_huv_path: list[str], output_huv_path: str, num_raster: int, max_velocity: float, file) -> None:
    """
    Process HUV data to an image."
    """
    height_data = load_terrain_data(input_huv_path[0])
    norm_height_data, min_height, max_height = process_terrain_data(height_data)

    u_data = load_terrain_data(input_huv_path[1])
    v_data = load_terrain_data(input_huv_path[2])
    norm_u_data, norm_v_data, min_u, max_u, min_v, max_v = process_uv_data(u_data, v_data, max_velocity)
    
    file.write(f"{num_raster}\t{min_height}\t{max_height}\t{min_u}\t{max_u}\t{min_v}\t{max_v}\n")

    image = create_huv_rgba_image(norm_height_data, norm_u_data, norm_v_data)
    save_image(image, output_huv_path)

def process_huv_to_image(input_path: str, output_path: str, max_velocity: float, start_time: int, delta_time: int, num_rasters: int, file) -> None:
    for num in range(num_rasters):
        num_raster = start_time + num * delta_time
        print(num_raster)
        input_huv_path = [input_path + 'h/depth_result_' + str(num_raster) + '.tif', 
                          input_path + 'u/u_result_' + str(num_raster) + '.tif', 
                          input_path + 'v/v_result_' + str(num_raster) + '.tif']
        output_huv_path = output_path + 'huv_' + str(num_raster) + '.png'
        encode_huv_to_image(input_huv_path, output_huv_path, num_raster, max_velocity, file)

def process_huv_to_image_from_datasets_optimized(datasets: list, output_path: str, file_suffix: str = "") -> None:
    """
    优化的函数，使用GPU加速处理HUV数据从内存数据集到单个图像
    """
    print(f"开始GPU加速处理数据集...")
    import time
    start_time = time.time()
    
    # 从datasets中提取h, u, v数据
    height_data = datasets[0].GetRasterBand(1).ReadAsArray().astype(np.float32)
    u_data = datasets[1].GetRasterBand(1).ReadAsArray().astype(np.float32)
    v_data = datasets[2].GetRasterBand(1).ReadAsArray().astype(np.float32)

    # 使用GPU加速处理高度数据
    norm_height_data, _, _ = process_terrain_data(height_data)

    # 使用GPU加速处理UV数据
    norm_u_data, norm_v_data, _, _, _, _ = process_uv_data(u_data, v_data, max_velocity=5.0)

    # 使用GPU加速创建HUV图像
    image = create_huv_rgba_image(norm_height_data, norm_u_data, norm_v_data)

    # 保存图像，使用file_suffix作为文件名后缀
    if file_suffix:
        output_huv_path = output_path + f'huv_{file_suffix}.png'
    else:
        output_huv_path = output_path + 'huv.png'
    save_image(image, output_huv_path)
    
    print(f"GPU加速处理完成，耗时: {time.time() - start_time:.2f} 秒")

def process_huv_to_image_from_datasets(datasets: list, output_path: str, file_suffix: str = "") -> None:
    """
    保持向后兼容的函数，调用优化版本
    """
    process_huv_to_image_from_datasets_optimized(datasets, output_path, file_suffix)

def process_huv_to_image(input_path: str, output_path: str, max_velocity: float, start_time: int, delta_time: int, num_rasters: int, file) -> None:
    for num in range(num_rasters):
        num_raster = start_time + num * delta_time
        print(num_raster)
        input_huv_path = [input_path + 'h/depth_result_' + str(num_raster) + '.tif', 
                          input_path + 'u/u_result_' + str(num_raster) + '.tif', 
                          input_path + 'v/v_result_' + str(num_raster) + '.tif']
        output_huv_path = output_path + 'huv_' + str(num_raster) + '.png'
        encode_huv_to_image(input_huv_path, output_huv_path, num_raster, max_velocity, file)



if __name__ == '__main__':
    # with open ('Textures/dem.txt', 'w') as file:
    #     process_dem_to_image('Rasters/DEM.tif', 'Textures/DEM.png', file)
    with open ('./Textures/huv.txt', 'w') as file:
        process_huv_to_image('./Rasters/', './Textures/', 5.0, 0, 1, 5, file)