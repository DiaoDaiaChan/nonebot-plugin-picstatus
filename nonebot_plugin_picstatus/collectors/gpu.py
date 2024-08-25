from dataclasses import dataclass
from typing import Optional, List
import pynvml
from nonebot import logger
from . import first_time_collector, periodic_collector


@dataclass
class GpuInfo:
    name: Optional[str]
    device_id: Optional[str]
    load: Optional[int]
    total_memory: Optional[int]
    free_memory: Optional[int]
    used_memory: Optional[int]
    temperature: Optional[float]
    driver_mode: Optional[str]
    percent: Optional[int]
    fan_speed: Optional[int]
    power_usage: Optional[int]
    performance_state: Optional[int]
    ecc_mode: Optional[str]


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
        else:
            return "TCC"
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
            load = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = int(mem_info.total / 1024 ** 2)
            free_memory = int(mem_info.free / 1024 ** 2)
            used_memory = int(mem_info.used / 1024 ** 2)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            driver_mode = _get_driver_mode(handle)
            percent = int(used_memory / total_memory * 100)

            fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            power_usage = int(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)
            performance_state = pynvml.nvmlDeviceGetPerformanceState(handle)
            try:
                pynvml.nvmlDeviceGetEccMode(handle)
            except pynvml.NVMLError_NotSupported:
                ecc_mode = "不支持"
            else:
                ecc_mode = "启动" if pynvml.nvmlDeviceGetEccMode(handle) else "关闭"

            gpu_info = GpuInfo(
                name=name,
                device_id=str(i),
                load=load,
                total_memory=total_memory,
                free_memory=free_memory,
                used_memory=used_memory,
                temperature=temperature,
                driver_mode=driver_mode,
                percent=percent,
                fan_speed=fan_speed,
                power_usage=power_usage,
                performance_state=performance_state,
                ecc_mode=ecc_mode
            )
            gpu_infos.append(gpu_info)
    except Exception:
        logger.exception("Error when collecting GPU info")
        # Handle the case where the error occurs before any GPUs have been added
        gpu_infos.append(
            GpuInfo(
                name=None,
                device_id=None,
                load=None,
                total_memory=None,
                free_memory=None,
                used_memory=None,
                temperature=None,
                driver_mode=None,
                percent=None,
                fan_speed=None,
                power_usage=None,
                performance_state=None,
                ecc_mode=None
            )
        )
    finally:
        shutdown_nvml()
    return gpu_infos


@periodic_collector()
async def render_gpu_info():
    gpu_data = await get_gpu_info()
    return gpu_data
