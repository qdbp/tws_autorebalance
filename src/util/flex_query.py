from io import StringIO
from time import sleep
from typing import Literal
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree.ElementTree import ParseError, fromstring

import pandas as pd
from pandas import DataFrame
from py9lib.io_ import TransientError, retry

from src import PROJECT_ROOT, config
from src.model import TZ_EASTERN_STR

BASE_URL = "https://gdcdyn.interactivebrokers.com"
QUERY_URL = BASE_URL + "/Universal/servlet/FlexStatementService.SendRequest"


Query = Literal["trades", "lots"]

DT_HP = "HoldingPeriodDateTime"
DT_O = "OpenDateTime"
DT_Rz = "WhenRealized"
DT_rO = "WhenReopened"
P_CB = "CostBasisPrice"
ID_ACCT = "ClientAccountID"
ID_OO = "OriginatingOrderID"
ID_OTx = "OriginatingTransactionID"
ID_O = "OrderID"
ID_Tr = "TradeId"
ID_Tx = "TransactionID"
CODE = "Code"
QTY = "Quantity"
SYM = "Symbol"

DTYPE = {
    CODE: str,
    ID_OTx: str,
    ID_Tx: str,
    ID_OO: str,
    ID_O: str,
    ID_Tr: str,
    "IBOrderID": str,
    "Open/CloseIndicator": str,
    "Notes/Codes": str,
}

FILLNA = {CODE: ""}

DATE_COLS = {DT_O, DT_HP, DT_Rz, DT_rO}


@retry((TransientError,), backoff_start=5.0, backoff_rate=5.0, max_retries=10)
def load_raw_query(query: Query) -> StringIO:

    cfg = config()["flex_query"]
    params = {"t": cfg["token"], "q": cfg[query], "v": "3"}

    query_url = f"{QUERY_URL}?{urlencode(params)}"
    raw = urlopen(query_url).read().decode("utf-8")
    tree = fromstring(raw)

    status = tree.find(".//Status")
    if not (status is not None and status.text == "Success"):
        raise IOError(
            f"Flex service returned error: "  # type: ignore
            f"{tree.find('.//ErrorMessage').text}"
        )

    flex_base: str = tree.find(".//Url").text  # type: ignore
    code = tree.find(".//ReferenceCode").text  # type: ignore
    payload_url = flex_base + "?" + urlencode(dict(t=cfg["token"], q=code))

    while True:
        raw_payload = urlopen(payload_url).read().decode("utf-8")
        try:
            tree = fromstring(raw_payload)
            payload_status: str = tree.find("code").text  # type: ignore
            if "generation in progress" in payload_status:
                sleep(1)
                continue
        except ParseError:
            break

    return StringIO(raw_payload)


def parse_query(raw: StringIO, query: Query) -> DataFrame:

    # filthy hack to get rid of duplicate headers...
    deduped = StringIO()
    pd.read_csv(raw, header=None).drop_duplicates().T.set_index(
        0, drop=True
    ).T.to_csv(deduped, index=False)

    deduped.seek(0)
    df = pd.read_csv(deduped, dtype=DTYPE).reset_index(drop=True)
    df.fillna(FILLNA, inplace=True)

    for col in df.columns.intersection(DATE_COLS):
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(TZ_EASTERN_STR)

    for garbage in ["index", "Unnamed:"]:
        bad_cols = df.filter(garbage)
        for col in bad_cols:
            df.drop(col, inplace=True, axis=1)

    return df


QUERY_CACHE_DIR = PROJECT_ROOT / "data" / "queries"


def load_query(query: Query, *, reload: bool) -> DataFrame:

    QUERY_CACHE_DIR.mkdir(exist_ok=True, parents=True)
    query_path = QUERY_CACHE_DIR.joinpath(f"{query}.csv")

    if query_path.exists() and not reload:
        with query_path.open("r") as f:
            raw = StringIO(f.read())
    else:
        raw = load_raw_query(query)
        with query_path.open("w") as f:
            f.write(raw.getvalue())

    out = parse_query(raw, query)
    return out
