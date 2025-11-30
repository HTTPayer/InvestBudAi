"""
Simple file-based cache with TTL (time-to-live).
"""
import os
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Callable


class Cache:
    """Simple file-based cache with TTL."""

    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.pkl"

    def _get_metadata_path(self, key: str) -> Path:
        """Get metadata file path for a key."""
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.meta.json"

    def get(self, key: str, ttl_seconds: int = 86400) -> Optional[Any]:
        """
        Get cached value if it exists and is fresh.

        Args:
            key: Cache key
            ttl_seconds: Time-to-live in seconds (default: 86400 = 24 hours)

        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)

        if not cache_path.exists() or not meta_path.exists():
            return None

        # Check if cache is still fresh
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            cached_at = datetime.fromisoformat(metadata['cached_at'])
            expires_at = cached_at + timedelta(seconds=ttl_seconds)

            if datetime.utcnow() > expires_at:
                print(f"[CACHE] Expired: {key} (cached at {cached_at})")
                return None

            # Load cached data
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            print(f"[CACHE] Hit: {key} (age: {datetime.utcnow() - cached_at})")
            return data

        except Exception as e:
            print(f"[CACHE] Error reading cache for {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        """
        Cache a value.

        Args:
            key: Cache key
            value: Value to cache (must be picklable)
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)

        try:
            # Save data
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)

            # Save metadata
            metadata = {
                'cached_at': datetime.utcnow().isoformat(),
                'key': key
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)

            print(f"[CACHE] Set: {key}")

        except Exception as e:
            print(f"[CACHE] Error writing cache for {key}: {e}")

    def get_or_set(self, key: str, fetch_fn: Callable[[], Any], ttl_seconds: int = 86400) -> Any:
        """
        Get cached value or fetch and cache it.

        Args:
            key: Cache key
            fetch_fn: Function to call if cache miss
            ttl_seconds: Time-to-live in seconds (default: 86400 = 24 hours)

        Returns:
            Cached or freshly fetched value
        """
        # Try to get from cache
        cached = self.get(key, ttl_seconds)
        if cached is not None:
            return cached

        # Cache miss - fetch and cache
        print(f"[CACHE] Miss: {key} - fetching fresh data...")
        value = fetch_fn()
        self.set(key, value)
        return value

    def clear(self, key: str) -> None:
        """Clear cache for a specific key."""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)

        cache_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        print(f"[CACHE] Cleared: {key}")

    def clear_all(self) -> None:
        """Clear all cache files."""
        for file in self.cache_dir.glob("*"):
            file.unlink()
        print(f"[CACHE] Cleared all cache files")


# Global cache instance
_cache = Cache(cache_dir="cache")


def get_cache() -> Cache:
    """Get global cache instance."""
    return _cache
