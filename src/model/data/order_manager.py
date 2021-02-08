from dataclasses import dataclass, field
from enum import Enum, auto
from time import monotonic
from typing import Optional, Union

from ibapi.order import Order

from src.model import Acct
from src.model.data import SimpleContract
from src.security.bounds import Policy
from src.util.format import pp_order


class OMState(Enum):
    ENTERED = auto()
    TRANSMITTED = auto()
    COOLOFF = auto()


@dataclass()
class OMRecord:
    oid: int
    sc: SimpleContract
    order: Order
    state: OMState
    touched: float = field(default_factory=monotonic)

    @property
    def acct(self) -> Acct:
        return Acct(self.order.account)


@dataclass()
class OMTombstone:
    acct: Acct
    sc: SimpleContract
    touched: float = field(default_factory=monotonic)


class OrderManager:
    """
    A bookkeeping ledger for orders. The duties of this class are to:

        - maintain an association between TWS order IDs and the associated
            contract
        - maintain an association between the TWS order ID and the Order sent
        - to maintain a "touched time" for each contract to guard against
            duplicated orders
        - to keep these ledgers consistent by encapsulation

    It operates as a state machine with the following allowed transitions:

    UNKNOWN <-> ENTERED -> TRANSMITTED -> COOLING OFF
        ^------------------------------------|

    An UNKNOWN order is one that is not tracked by the manager. This is the
    initial state of all orders. Untransmitted orders may be deleted
    to revert to this state, and cooling off orders enter this state after
    the cool-off period expires.

    An ENTERED order is one that is recorded in the ledger but not live in TWS.
    This is a clear distinction when orders are sent to TWS with transmit=False,
    but must be carefully managed by the caller when this is not the case. An
    ENTERED order can be removed without triggering a cooloff, for example to
    replace an untransmitted order with one of a different method or quantity.

    A TRANSMITTED contract has live order in TWS. It cannot be cleared. It must
    be finalized instead, which will trigger the cooloff period.

    A contract in COOLOFF is not active or staged in TWS, but cannot be entered
    because it was active too recently. This debounce mechanism is there to
    safeguard against duplicate orders. An order in COOLOFF will (as far as the
    caller is concerned) automatically transition to UNKNOWN when its holding
    time expires.

    NB. This ledger is not integrated with an EClient or EWrapper, and is purely
    a side-accounting tool. It cannot transmit, cancel, or modify actual TWS
    orders.
    """

    def __init__(self) -> None:
        self._records: dict[int, Union[OMRecord, OMTombstone]] = {}

    def reap(self, oid: int) -> None:
        if (
            (rec := self._records.get(oid))
            and (isinstance(rec, OMTombstone) or rec.state == OMState.COOLOFF)
            and rec.touched + Policy.ORDER_COOLOFF < monotonic()
        ):
            del self._records[oid]

    def find_record(
        self, acct: Acct, sc: SimpleContract
    ) -> Optional[Union[OMRecord, OMTombstone]]:
        for oid in list(self._records.keys()):
            self.reap(oid)
        for oid, rec in self._records.items():
            if rec.acct == acct and rec.sc == sc:
                return rec
        return None

    def get_record(self, oid: int) -> Optional[Union[OMRecord, OMTombstone]]:
        self.reap(oid)
        return self._records.get(oid)

    def enter_order(self, oid: int, sc: SimpleContract, order: Order) -> bool:
        rec = self.find_record(acct=order.account, sc=sc)
        if rec is not None:
            return False
        else:
            self._records[oid] = OMRecord(
                oid=oid, sc=sc, order=order, state=OMState.ENTERED
            )
            return True

    def clear_untransmitted(
        self, acct: Acct, sc: SimpleContract
    ) -> Optional[OMRecord]:
        """
        Will clear an untransmitted order for `sc`, if one exists.

        If an order was cleared, will return its record. Else returns
        None.
        """
        rec = self.find_record(acct, sc)
        if isinstance(rec, OMRecord) and rec.state == OMState.ENTERED:
            del self._records[rec.oid]
            return rec
        return None

    def transmit_order(self, oid: int) -> bool:
        rec = self.get_record(oid)
        if (
            rec is None
            or isinstance(rec, OMTombstone)
            or rec.state != OMState.ENTERED
        ):
            return False
        rec.state = OMState.TRANSMITTED
        rec.touched = monotonic()
        return True

    def finalize_order(self, oid: int) -> bool:
        rec = self.get_record(oid)
        if (
            rec is None
            or isinstance(rec, OMTombstone)
            or rec.state != OMState.TRANSMITTED
        ):
            return False
        rec.state = OMState.COOLOFF
        rec.touched = monotonic()
        return True

    def format_book(self) -> str:
        out = ""
        for oid, rec in self._records.items():
            sc = rec.sc
            if isinstance(rec, OMTombstone):
                state = OMState.COOLOFF
                order = None
            else:
                state = rec.state
                order = rec.order
            msg = f"Order Book: {sc.symbol} = {state}"
            if state == OMState.ENTERED or state == OMState.TRANSMITTED:
                msg += f": {pp_order(sc.contract, order)}"
            out += msg + "\n"
        return out

    def force_cool(self, acct: Acct, sc: SimpleContract) -> None:
        """
        Forces an account + sc pair into cooloff, tracked or not.
        """
        rec = self.find_record(acct, sc)
        if rec is None:
            # dummy non-conflicting key
            self._records[-int(monotonic())] = OMTombstone(sc=sc, acct=acct)
        else:
            rec.touched = monotonic()
