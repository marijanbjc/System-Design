"""Token bucket и бюджетный предохранитель перед каждым обращением к LLM.

Ограничитель нужен не ради экономии токенов, а чтобы не сделать провайдера
неработоспособным: тикет у оператора стоит ~150 ₽, вызов модели — доли рубля.
Дневной бюджет — предохранитель от бага и петли ретраев, а не инструмент экономии.
"""

import time

import redis

from app.config import get_settings

BUCKET_KEY = "llm:bucket"
BUDGET_KEY_PREFIX = "llm:budget:"

# Атомарная выдача разрешения: сначала пополняем ведро по прошедшему времени,
# затем тратим одно, если есть. Lua нужен, чтобы это не разъезжалось между репликами.
_TAKE_TOKEN_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

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
redis.call('EXPIRE', key, 3600)
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
            keys=[BUCKET_KEY],
            args=[settings.llm_rate_limit_rps, settings.llm_rate_limit_burst, time.time()],
        )
        return bool(int(allowed))

    def acquire(self, timeout: float = 10.0, poll: float = 0.05) -> bool:
        """Подождать разрешения до таймаута. Очередь притормаживает вход сама собой."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.try_acquire():
                return True
            time.sleep(poll)
        return False


class BudgetBreaker:
    """Дневной бюджет токенов. Сработал — авто-генерация выключается, тикеты идут к людям."""

    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    def _key(self) -> str:
        return f"{BUDGET_KEY_PREFIX}{time.strftime('%Y-%m-%d')}"

    def spent(self) -> int:
        """Сколько токенов израсходовано за сегодня."""
        value = self._client.get(self._key())
        return int(value) if value else 0

    def charge(self, tokens: int) -> None:
        """Списать израсходованные токены."""
        key = self._key()
        pipe = self._client.pipeline()
        pipe.incrby(key, tokens)
        pipe.expire(key, 172_800)
        pipe.execute()

    def is_open(self) -> bool:
        """True, когда предохранитель разомкнут и звать провайдера больше нельзя."""
        return self.spent() >= get_settings().llm_daily_token_budget
