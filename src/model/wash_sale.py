from datetime import date, timedelta

from src.model.trade import Trade


def wash_window(trade: Trade) -> tuple[date, date]:
    td = trade.dt.date()
    return td - timedelta(days=30), td + timedelta(days=30)


def is_in_wash_window(d: date, trade: Trade) -> bool:
    lb, ub = wash_window(trade)
    return lb <= d <= ub
