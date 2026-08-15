"""Token bucket перед каждым обращением к LLM.

Ограничитель нужен не ради экономии токенов, а чтобы не сделать провайдера
неработоспособным: тикет у оператора стоит ~150 ₽, вызов модели — доли рубля.
Бюджетный предохранитель описан в architecture.md §9 как целевой механизм и в PoC
не реализован: здесь остаётся минимальная деградация «вызов упал → тикет к оператору».
"""

import time

import redis

from app.config import get_settings

# Атомарная выдача разрешения: сначала пополняем ведро по прошедшему времени,
# затем тратим одно, если есть. Lua нужен, чтобы это не разъезжалось между репликами.
_TAKE_TOKEN_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil then
  tokens = burst
  ts = now
end

tokens = math.min(burst, tokens + (now - ts) * rate)
local allowed = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, ttl)
return allowed
"""


class RateLimiter:
    """Общий на все реплики воркера: лимит глобальный, а не на процесс."""

    def __init__(self, client: redis.Redis) -> None:
        self._client = client
        self._script = client.register_script(_TAKE_TOKEN_LUA)

    def try_acquire(self) -> bool:
        """Взять одно разрешение. False означает «подождать», а не «выбросить задачу»."""
        settings = get_settings()
        allowed = self._script(
            keys=[settings.rate_limit_bucket_key],
            args=[
                settings.llm_rate_limit_rps,
                settings.llm_rate_limit_burst,
                time.time(),
                settings.llm_bucket_ttl_seconds,
            ],
        )
        return bool(int(allowed))

    def acquire(self, timeout: float) -> bool:
        """Подождать разрешения до таймаута. Очередь притормаживает вход сама собой."""
        poll = get_settings().llm_acquire_poll_seconds
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.try_acquire():
                return True
            time.sleep(poll)
        return False
