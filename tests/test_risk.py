import numpy as np
import pytest

from marketzero.config import Config
from marketzero.risk import RiskGovernor


def test_kill_switch_on_drawdown():
    gov = RiskGovernor(Config())
    ok, _ = gov.check({"drawdown": 0.19, "daily_loss": 0.0})
    assert ok
    ok, reasons = gov.check({"drawdown": 0.21, "daily_loss": 0.0})
    assert not ok and any("drawdown" in r for r in reasons)


def test_kill_switch_on_daily_loss():
    gov = RiskGovernor(Config())
    ok, reasons = gov.check({"drawdown": 0.0, "daily_loss": 0.09})
    assert not ok and any("daily loss" in r for r in reasons)


def test_halt_is_sticky_until_rearmed():
    gov = RiskGovernor(Config())
    gov.check({"drawdown": 0.5, "daily_loss": 0.0})
    ok, _ = gov.check({"drawdown": 0.0, "daily_loss": 0.0})
    assert not ok                       # stays halted even when stats look fine
    gov.re_arm()
    ok, _ = gov.check({"drawdown": 0.0, "daily_loss": 0.0})
    assert ok


def test_vol_target_scaling():
    cfg = Config()
    cfg.risk_vol_target = 0.15
    gov = RiskGovernor(cfg)
    for _ in range(200):                # pump variance up (~high annualized vol)
        gov.observe_return(0.05)
    assert gov.position_scale(1) < 0.5  # shrinks a chatty strategy hard


def test_no_vol_target_is_full_size():
    cfg = Config()
    cfg.risk_vol_target = None
    gov = RiskGovernor(cfg)
    assert gov.position_scale(1) == 1.0
