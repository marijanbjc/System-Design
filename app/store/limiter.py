"""Token-bucket rate limiter and daily budget breaker in front of every LLM call.

The limiter exists to protect the provider from us, not to save tokens: an operator
ticket costs ~150 RUB and an LLM call fractions of one. The daily budget is a fuse
against a bug or a retry loop, not a cost optimization.
"""

import time

import redis

from app.config import get_settings

BUCKET_KEY = "llm:bucket"
BUDGET_KEY_PREFIX = "llm:budget:"

# Atomic take-a-permit: refill by elapsed time, then spend one if available.
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
    """Shared across all worker replicas: the limit is global, not per process."""

    def __init__(self, client: redis.Redis) -> None:
        self._client = client
        self._script = client.register_script(_TAKE_TOKEN_LUA)

    def try_acquire(self) -> bool:
        """Take one permit. False means the caller must wait, not drop the task."""
        settings = get_settings()
        allowed = self._script(
            keys=[BUCKET_KEY],
            args=[settings.llm_rate_limit_rps, settings.llm_rate_limit_burst, time.time()],
        )
        return bool(int(allowed))

    def acquire(self, timeout: float = 10.0, poll: float = 0.05) -> bool:
        """Block until a permit is free or the timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.try_acquire():
                return True
            time.sleep(poll)
        return False


class BudgetBreaker:
    """Daily token budget. Once tripped, auto-generation stops and tickets go to humans."""

    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    def _key(self) -> str:
        return f"{BUDGET_KEY_PREFIX}{time.strftime('%Y-%m-%d')}"

    def spent(self) -> int:
        value = self._client.get(self._key())
        return int(value) if value else 0

    def charge(self, tokens: int) -> None:
        key = self._key()
        pipe = self._client.pipeline()
        pipe.incrby(key, tokens)
        pipe.expire(key, 172_800)
        pipe.execute()

    def is_open(self) -> bool:
        """True when the fuse has blown and we must stop calling the provider."""
        return self.spent() >= get_settings().llm_daily_token_budget
