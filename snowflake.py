import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import time
from typing import Optional
 
__all__ = ("Snowflake", "SnowflakeGenerator")
 
MAX_INSTANCE = 0b1111111111  # 分散式系統最多 1023 個節點
MAX_SEQ = 0b111111111111  # 每毫秒能生成的 4095 ID 數量
MAX_TS = 0b11111111111111111111111111111111111111111  # 大約等於 69.73 年
 
 
@dataclass(frozen=True)
class Snowflake:
    timestamp: int
    instance: int
    epoch: int = 0  # 起始時間點（epoch）
    seq: int = 0
 
    def __post_init__(self):
        if self.epoch < 0:
            raise ValueError(f"epoch must be greater than 0!")
 
        if self.timestamp < 0 or self.timestamp > MAX_TS:
            raise ValueError(
                f"timestamp must not be negative and must be less than {MAX_TS}!"
            )
 
        if self.instance < 0 or self.instance > MAX_INSTANCE:
            raise ValueError(
                f"instance must not be negative and must be less than {MAX_INSTANCE}!"
            )
 
        if self.seq < 0 or self.seq > MAX_SEQ:
            raise ValueError(
                f"seq must not be negative and must be less than {MAX_SEQ}!"
            )
 
    @classmethod
    def parse(cls, snowflake: int, epoch: int = 0) -> "Snowflake":
        return cls(
            epoch=epoch,
            timestamp=snowflake >> 22,
            instance=snowflake >> 12 & MAX_INSTANCE,
            seq=snowflake & MAX_SEQ,
        )
 
    @property
    def milliseconds(self) -> int:
        return self.timestamp + self.epoch
 
    @property
    def seconds(self) -> float:
        return self.milliseconds / 1000
 
    @property
    def datetime(self) -> datetime:
        return datetime.utcfromtimestamp(self.seconds)
 
    @property
    def timedelta(self) -> timedelta:
        return timedelta(milliseconds=self.epoch)
 
    @property
    def value(self) -> int:
        return self.timestamp << 22 | self.instance << 12 | self.seq
 
    def __int__(self) -> int:
        return self.value
 
 
class SnowflakeGenerator:
    def __init__(
        self,
        instance: int,
        *,
        seq: int = 0,
        epoch: int = 0,
        timestamp: Optional[int] = None,
    ):
        current = (time() * 1000.0).__int__()
 
        if current - epoch >= MAX_TS:
            raise OverflowError(
                f"The maximum current timestamp has been reached in selected epoch,"
                f"so Snowflake cannot generate more IDs!"
            )
 
        timestamp = timestamp or current
 
        if timestamp < 0 or timestamp > current:
            raise ValueError(
                f"timestamp must not be negative and must be less than {current}!"
            )
 
        if epoch < 0 or epoch > current:
            raise ValueError(
                f"epoch must not be negative and must be lower than current time {current}!"
            )
 
        self._epo = epoch
        self._ts = timestamp - self._epo
 
        if instance < 0 or instance > MAX_INSTANCE:
            raise ValueError(
                f"instance must not be negative and must be less than {MAX_INSTANCE}!"
            )
 
        if seq < 0 or seq > MAX_SEQ:
            raise ValueError(
                f"seq must not be negative and must be less than {MAX_SEQ}!"
            )
 
        self._inf = instance << 12
        self._seq = seq
 
    @classmethod
    def from_snowflake(cls, sf: Snowflake) -> "SnowflakeGenerator":
        return cls(sf.instance, seq=sf.seq, epoch=sf.epoch, timestamp=sf.timestamp)
 
    @property
    def epoch(self) -> int:
        return self._epo
 
    def __iter__(self):
        return self
 
    def __next__(self) -> str:
        current = (time() * 1000.0).__int__() - self._epo
 
        if current >= MAX_TS:
            raise OverflowError(
                f"The maximum current timestamp has been reached in selected epoch,"
                f"so Snowflake cannot generate more IDs!"
            )
 
        if self._ts == current:
            if self._seq == MAX_SEQ:
                raise OverflowError(
                    f"The maximum current sequence has been reached in selected epoch,"
                    f"so Snowflake cannot generate more IDs!"
                )
            self._seq += 1
        else:
            self._seq = 0
 
        self._ts = current
        return str(self._ts << 22 | self._inf | self._seq)
 
    def __call__(self) -> str:
        return self.__next__()
 
snowflake_id = SnowflakeGenerator(random.randint(1, 1023))