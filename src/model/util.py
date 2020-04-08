from ibapi.contract import Contract
from ibapi.order import Order


def pp_order(nc: Contract, order: Order):
    return f"{order.action} {order.totalQuantity} {nc.symbol} ({order.orderType})"
