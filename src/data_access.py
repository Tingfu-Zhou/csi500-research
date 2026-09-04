from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from src.date_utils import parse_date_series


class WindowViolation(RuntimeError):
    """Raised when design-stage code requests sealed out-of-sample data."""


class ResearchDataAccess:
    """The only allowed gateway to Qlib and date-bearing Parquet data."""

    def __init__(self, project_root: Path, config_path: Path | None = None):
        self.root = Path(project_root).resolve()
        self.config_path = config_path or self.root / "config/config.yaml"
        with self.config_path.open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        self.design_end = pd.Timestamp(self.config["windows"]["design_end"])
        self.marker_path = self.root / self.config["data"]["window_marker"]
        self._qlib_ready = False

    @property
    def mode(self) -> str:
        if not self.marker_path.exists():
            return "design"
        payload = json.loads(self.marker_path.read_text(encoding="utf-8"))
        return payload.get("mode", "design")

    def assert_window(self, start_date=None, end_date=None) -> None:
        del start_date
        if end_date is None:
            raise WindowViolation("所有数据请求都必须显式给出 end_date")
        end = pd.Timestamp(end_date).normalize()
        if self.mode == "design" and end >= pd.Timestamp("2019-01-01"):
            raise WindowViolation(
                f"设计窗口已封存：end_date={end.date()}，必须早于 2019-01-01"
            )

    def _init_qlib(self) -> None:
        if self._qlib_ready:
            return
        import qlib

        qlib.init(provider_uri=self.config["data"]["qlib_provider_uri"])
        self._qlib_ready = True

    def features(
        self,
        instruments: Iterable[str],
        fields: list[str],
        start_date,
        end_date,
        freq: str = "day",
    ) -> pd.DataFrame:
        self.assert_window(start_date, end_date)
        self._init_qlib()
        from qlib.data import D

        return D.features(
            list(instruments),
            fields,
            start_time=str(pd.Timestamp(start_date).date()),
            end_time=str(pd.Timestamp(end_date).date()),
            freq=freq,
        )

    def instruments(self, market: str, start_date, end_date, as_list=False):
        self.assert_window(start_date, end_date)
        self._init_qlib()
        from qlib.data import D

        return D.list_instruments(
            D.instruments(market),
            start_time=str(pd.Timestamp(start_date).date()),
            end_time=str(pd.Timestamp(end_date).date()),
            as_list=as_list,
        )

    def read_parquet(
        self,
        relative_path: str | Path,
        *,
        date_col: str,
        start_date=None,
        end_date,
    ) -> pd.DataFrame:
        self.assert_window(start_date, end_date)
        path = (self.root / relative_path).resolve()
        sealed_root = (self.root / "data/_sealed").resolve()
        if self.mode == "design" and (path == sealed_root or sealed_root in path.parents):
            raise WindowViolation(f"设计阶段禁止读取 sealed 路径：{path}")
        frame = pd.read_parquet(path)
        dates = parse_date_series(frame[date_col], errors="raise")
        mask = dates <= pd.Timestamp(end_date)
        if start_date is not None:
            mask &= dates >= pd.Timestamp(start_date)
        return frame.loc[mask].copy()
