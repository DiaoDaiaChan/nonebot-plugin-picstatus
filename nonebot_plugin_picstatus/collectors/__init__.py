import asyncio
import importlib
import time
from abc import abstractmethod
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import override

from cookit import SizedList
from nonebot import logger
from nonebot_plugin_apscheduler import scheduler

from ..config import config

T = TypeVar("T")
TC = TypeVar("TC", bound="Collector")
TCF = TypeVar("TCF", bound=Callable[[], Awaitable[Any]])
R = TypeVar("R")

Undefined = type("Undefined", (), {})


class SkipCollectError(Exception):
    pass


class Collector(Generic[T]):
    @abstractmethod
    async def get(self) -> T: ...


class BaseNormalCollector(Generic[T], Collector[T]):
    def __init__(self) -> None:
        super().__init__()


class BaseFirstTimeCollector(Generic[T], Collector[T]):
    def __init__(self) -> None:
        super().__init__()
        self._cached: Union[T, Undefined] = Undefined()

    async def get(self) -> T:
        if not isinstance(self._cached, Undefined):
            return self._cached
        data = await self.get()
        self._cached = data
        return data


class BasePeriodicCollector(Generic[T], Collector[SizedList[T]]):
    def __init__(self, size: int = config.ps_default_collect_cache_size) -> None:
        super().__init__()
        self.data = SizedList[T](size=size)

    async def get(self) -> SizedList[T]:
        return self.data

    @abstractmethod
    async def _do_collect(self) -> T: ...

    async def collect(self):
        try:
            data = await self._do_collect()
        except SkipCollectError:
            return
        except Exception:
            logger.exception("Error occurred while collecting data")
        else:
            self.data.append(data)


registered_collectors: Dict[str, Type[Collector]] = {}
enabled_collectors: Dict[str, Collector] = {}


def collector(name: str):
    def deco(cls: Type[TC]) -> Type[TC]:
        if name in registered_collectors:
            raise ValueError(f"Collector {name} already exists")
        registered_collectors[name] = cls
        return cls

    return deco


def enable_collector(name: str):
    cls = registered_collectors[name]
    if issubclass(cls, BasePeriodicCollector) and name in config.ps_collect_cache_size:
        instance = cls(size=config.ps_collect_cache_size[name])
    else:
        instance = cls()
    enabled_collectors[name] = instance


def functional_collector(cls: Type[Collector], name: Optional[str] = None):
    def deco(func: TCF) -> TCF:
        collector_name = name or func.__name__
        if not name:
            raise ValueError("name must be provided")

        class Collector(cls):
            async def get(self) -> Any:
                return await func()

        collector(collector_name)(Collector)
        return func

    return deco


def normal_collector(name: Optional[str] = None):
    return functional_collector(BaseNormalCollector, name)


def first_time_collector(name: Optional[str] = None):
    return functional_collector(BaseFirstTimeCollector, name)


def periodic_collector(name: Optional[str] = None):
    return functional_collector(BasePeriodicCollector, name)


class TimeBasedCounterCollector(Generic[T, R], BasePeriodicCollector[R]):
    def __init__(self, size: int = config.ps_default_collect_cache_size) -> None:
        super().__init__(size)
        self.last_obj: Union[Undefined, T] = Undefined()
        self.last_time: float = 0

    @abstractmethod
    async def _calc(self, past: T, now: T, time_passed: float) -> R: ...

    @abstractmethod
    async def _get_obj(self) -> T: ...

    @override
    async def _do_collect(self) -> R:
        past = self.last_obj
        past_time = self.last_time
        time_now = time.time()
        time_passed = time_now - past_time

        self.last_time = time_now
        self.last_obj = await self._get_obj()
        if isinstance(past, Undefined):
            raise SkipCollectError
        return await self._calc(past, self.last_obj, time_passed)


@scheduler.scheduled_job("interval", seconds=config.ps_collect_interval)
async def _():
    await asyncio.gather(
        *(
            x.collect()
            for x in enabled_collectors.values()
            if isinstance(x, BasePeriodicCollector)
        ),
    )


async def collect_all() -> Dict[str, Any]:
    async def get(name: str):
        return name, await enabled_collectors[name].get()

    res = await asyncio.gather(*(get(name) for name in enabled_collectors))
    return dict(res)


def load_collectors():
    for module in Path(__file__).parent.iterdir():
        if not module.name.startswith("_"):
            importlib.import_module(f".{module.stem}", __package__)
