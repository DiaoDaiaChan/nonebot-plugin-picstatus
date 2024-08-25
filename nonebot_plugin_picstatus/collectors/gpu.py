from dataclasses import dataclass
from typing import Optional, Tuple, List

import pynvml
from nonebot import logger

from . import first_time_collector, periodic_collector


@dataclass
class GpuInfo:
    name: Optional[str]
    device_id: Optional[str]
    total_memory: Optional[int]
    free_memory: Optional[int]
    used_memory: Optional[int]
    temperature: Optional[float]
    driver_mode: Optional[str]
    percent: Optional[float]


def init_nvml():
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as err:
        logger.exception("Failed to initialize NVML: %s", err)


def shutdown_nvml():
    try:
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as err:
        logger.exception("Failed to shutdown NVML: %s", err)



def _get_driver_mode(handle) -> Optional[str]:
    try:
        driver_model_current, _ = pynvml.nvmlDeviceGetDriverModel(handle)
        if driver_model_current == pynvml.NVML_DRIVER_WDDM:
            return "WDDM"
        elif driver_model_current == pynvml.NVML_DRIVER_TCC:
            return "TCC"
        else:
            return "Unknown"
    except pynvml.NVMLError as err:
        logger.exception("Error when getting GPU driver mode: %s", err)
        return None


async def get_gpu_info() -> List[GpuInfo]:
    init_nvml()
    gpu_infos = []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = int(mem_info.total / 1024 ** 2)  # Convert to MB
            free_memory = int(mem_info.free / 1024 ** 2)  # Convert to MB
            used_memory = int(mem_info.used / 1024 ** 2)  # Convert to MB
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            driver_mode = _get_driver_mode(handle)
            percent = used_memory / total_memory * 100

            gpu_info = GpuInfo(
                name=name,
                device_id=str(i),
                total_memory=total_memory,
                free_memory=free_memory,
                used_memory=used_memory,
                temperature=temperature,
                driver_mode=driver_mode,
                percent=percent
            )
            gpu_infos.append(gpu_info)
    except Exception:
        logger.exception("Error when collecting GPU info")
        # Handle the case where the error occurs before any GPUs have been added
        gpu_infos.append(
            GpuInfo(
                name=None,
                device_id=None,
                total_memory=None,
                free_memory=None,
                used_memory=None,
                temperature=None,
                driver_mode=None
            )
        )
    finally:
        shutdown_nvml()
    return gpu_infos


# 您的其他 GPU 信息收集器函数可以保持不变
# 它们将从 `get_gpu_info` 中的 GPU 信息中获取数据

# 示例：使用 `get_gpu_info` 获取信息并传递给模板
@periodic_collector()
async def render_gpu_info():
    gpu_data = await get_gpu_info()
    return gpu_data
