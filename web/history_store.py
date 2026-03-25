"""
Optional analysis history persistence.

This module keeps database integration isolated from the Flask app so the
runtime stays lightweight and continues working when no database is configured.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib import error, parse, request


LOGGER = logging.getLogger(__name__)


class NullHistoryStore:
    """No-op history backend used when persistence is not configured."""

    backend_name = "disabled"

    def is_enabled(self) -> bool:
        return False

    def save_many(self, records: list[dict[str, Any]]) -> None:
        return None

    def list_recent(self, limit: int = 12) -> list[dict[str, Any]]:
        return []


class SupabaseHistoryStore:
    """Persist lightweight analysis history rows through Supabase REST."""

    backend_name = "supabase"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        table_name: str = "analysis_history",
        timeout_seconds: float = 4.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.table_name = table_name
        self.timeout_seconds = timeout_seconds
        self.rest_url = f"{self.base_url}/rest/v1/{self.table_name}"

    def is_enabled(self) -> bool:
        return True

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json",
        }
        if not self.api_key.startswith("sb_secret_"):
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            headers.update(extra)
        return headers

    def save_many(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        payload = json.dumps(records).encode("utf-8")
        req = request.Request(
            self.rest_url,
            data=payload,
            headers=self._headers({"Prefer": "return=minimal"}),
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds):
                return None
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            LOGGER.warning("Supabase history save failed: %s %s", exc.code, body)
        except Exception as exc:
            LOGGER.warning("Supabase history save failed: %s", exc)

    def list_recent(self, limit: int = 12) -> list[dict[str, Any]]:
        query = parse.urlencode(
            {
                "select": (
                    "id,created_at,filename,verdict,verdict_short,ensemble_score,"
                    "input_preview_url,fft_spectrum_url,ela_heatmap_url,image_info,"
                    "detector_scores"
                ),
                "order": "created_at.desc",
                "limit": str(max(1, min(limit, 24))),
            }
        )
        req = request.Request(
            f"{self.rest_url}?{query}",
            headers=self._headers(),
            method="GET",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
            data = json.loads(payload)
            return data if isinstance(data, list) else []
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            LOGGER.warning("Supabase history fetch failed: %s %s", exc.code, body)
        except Exception as exc:
            LOGGER.warning("Supabase history fetch failed: %s", exc)
        return []


def build_history_store() -> NullHistoryStore | SupabaseHistoryStore:
    """Create the configured history backend, defaulting to no-op mode."""
    base_url = os.environ.get("SUPABASE_URL", "").strip()
    api_key = (
        os.environ.get("SUPABASE_SERVER_KEY", "").strip()
        or os.environ.get("SUPABASE_SECRET_KEY", "").strip()
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )
    table_name = os.environ.get("SUPABASE_HISTORY_TABLE", "analysis_history").strip()

    if not base_url or not api_key:
        return NullHistoryStore()

    return SupabaseHistoryStore(
        base_url=base_url,
        api_key=api_key,
        table_name=table_name or "analysis_history",
    )
