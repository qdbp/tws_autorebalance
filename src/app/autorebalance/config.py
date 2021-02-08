from __future__ import annotations

from dataclasses import Field, dataclass, fields, is_dataclass
from io import StringIO
from typing import Any, Optional

from src import config
from src.model import Acct
from src.model.data import Composition
from src.security.bounds import Policy


@dataclass(frozen=True)
class AutoRebalanceConfig:

    # orders
    order_timeout: int
    max_slippage: float
    rebalance_freq: float
    liveness_timeout: float
    armed: bool

    accounts: dict[Acct, AutoRebalanceAcctConfig]

    def __post_init__(self) -> None:
        for f in fields(self):
            assert getattr(self, f.name) is not None

    @classmethod
    def load(cls) -> AutoRebalanceConfig:

        raw_config = config()
        strategy = raw_config["strategy"]

        acct_configs = {
            acct_dict[acct]: AutoRebalanceAcctConfig.from_dict(acct_dict)
            for acct, acct_dict in strategy["accounts"].items()
        }

        return cls(
            # raw_config=raw_config,
            accounts=acct_configs,
            **raw_config["settings"],
        )

    def dump_config(self) -> str:
        out = StringIO()

        for field in fields(self):
            attr = getattr(self, field.name)
            print(f"{field} = {attr}", file=out)
            if is_dataclass(attr):
                for i_field in fields(attr):
                    print(
                        f"{i_field} = {getattr(attr, i_field.name)}", file=out
                    )

        return out.getvalue()


@dataclass(frozen=True)
class AutoRebalanceAcctConfig:
    acct: Acct

    # rebalancing eagerness
    misalloc_frac_coef: float
    misalloc_pvf_coef: float

    # composition
    composition: Composition
    goal_composition: Composition

    # margin
    margin: Optional[AutoRebalanceMarginConfig]

    def __post_init__(self) -> None:
        f: Field[Any]
        for f in fields(self):
            assert getattr(self, f.name) is not None

    @classmethod
    def from_dict(cls, acct_dict: dict[str, Any]) -> AutoRebalanceAcctConfig:
        """
        Constructs an account composition from an account sub-dict.
        """

        comp, target = Composition.parse_config_section(
            acct_dict.pop("composition")
        )

        return AutoRebalanceAcctConfig(
            composition=comp,
            goal_composition=target,
            margin=(
                AutoRebalanceMarginConfig(**acct_dict.pop("margin"))
                if "margin" in acct_dict
                else None
            ),
            **(acct_dict.pop("rebalance")),
            **acct_dict,
        )

    def dump_config(self) -> str:
        return "\n".join(
            f"{field.name}={getattr(self, field.name)}"
            for field in fields(self)
        )


@dataclass(frozen=True)
class AutoRebalanceMarginConfig:
    dd_reference_ath: float
    mu_at_ath: float
    dd_coef: float
    min_margin_req: float
    update_ath: bool

    def __post_init__(self) -> None:
        f: Field[Any]
        for f in fields(self):
            assert getattr(self, f.name) is not None
        Policy.ATH_MARGIN_USE.validate(self.mu_at_ath)
        Policy.DRAWDOWN_COEFFICIENT.validate(self.dd_coef)
        Policy.MARGIN_REQ.validate(self.min_margin_req)
