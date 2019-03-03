from __future__ import (
    division,
    print_function,
)

import csv
import os
import re
import requests
import shutil
import time

from collections import (
    defaultdict,
    namedtuple,
)
from contextlib import contextmanager
from datetime import date, datetime, timedelta, time as dt_time
from dateutil.parser import parse as dateparse
from dateutil.tz import tzutc, tzoffset
from io import StringIO
from pprint import pprint

from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

GDAX_API = 'https://api.gdax.com'

class CommonRow(NamedTuple):
    pair: str
    src: float
    dst: float
    site: str
    ts_orig: str
    ts_parsed: datetime

class CommonRowWUSD(NamedTuple):
    pair: str
    src: float
    dst: float
    site: str
    ts_parsed: datetime
    src_usd: Optional[float]
    dst_usd: Optional[float]

class Buy(NamedTuple):
    typ: str
    amt: float
    cost: float
    site: str
    ts_parsed: datetime

class Sell(NamedTuple):
    typ: str
    amt: float
    cost: float
    site: str
    ts_parsed: datetime

class CostBasis(NamedTuple):
    typ: str
    amt: float
    proceeds: float
    basis: float
    buy_ts: datetime
    sell_ts: datetime
    buy_site: str
    sell_site: str

Buys = List[Buy]
Sells = List[Sell]
BuyOrSellType = Union[Type[Buy], Type[Sell]]

def cost_basis(buys_sells):
    # type: (List[Union[Buy, Sell]]) -> Tuple[List[CostBasis], List[Buy]]
    """Returns cost basis of realized gains + leftover buys"""
    cbs = []
    buys = []

    buys_sells = sorted(buys_sells, key=lambda row: (row.ts_parsed, type(row).__name__))

    for bs in buys_sells:
        if isinstance(bs, Buy):
            buys.append(bs)
            continue
        assert isinstance(bs, Sell)
        sell = bs

        while sell.amt > 0:
            buy_idx = select_buy(sell, buys)
            assert buy_idx < len(buys), "Buy out of bounds. Sell={}".format(sell)
            buy = buys[buy_idx]
            assert buy.typ == sell.typ
            if buy.amt > sell.amt:
                # Split the buy
                basis = buy.cost * (sell.amt / buy.amt)
                buys[buy_idx] = buy._replace(amt=buy.amt - sell.amt, cost=buy.cost - basis)
                cbs.append(CostBasis(sell.typ, sell.amt, sell.cost, basis,
                                     buy.ts_parsed, sell.ts_parsed, buy.site, sell.site))

                if buys[buy_idx].cost <= 1e-6:
                    del buys[buy_idx]
                sell = sell._replace(amt=0, cost=0)
            else:
                # Split the sell
                assert buy.amt <= sell.amt
                partial_sell_cost = sell.cost * (buy.amt / sell.amt)
                cbs.append(CostBasis(sell.typ, buy.amt, partial_sell_cost, buy.cost,
                                     buy.ts_parsed, sell.ts_parsed, buy.site, sell.site))

                del buys[buy_idx]
                sell = sell._replace(amt=sell.amt - buy.amt, cost=sell.cost - partial_sell_cost)
                if sell.amt <= 1e-6:
                    sell = sell._replace(amt=0, cost=0)

    return cbs, buys

def select_buy(sell, buys):
    # type: (Sell, List[Buy]) -> int
    """Here's where it gets interesting. Given the set of unused buy, select
    one of those buys for cost basis, and remove from list of buys.

    This is where FIFO or LIFO or something more complex happens

    Return the idx into the buys"""
    assert buys == sorted(buys, key=lambda r: r.ts_parsed)

    def smart_lifo():
        # type: () -> int
        # If it's ST no matter what, take the shortest
        if sell.ts_parsed - buys[0].ts_parsed < timedelta(days=366):
            return len(buys) - 1
        # Otherwise take the shortest LT
        for idx, buy in reversed(list(enumerate(buys))):
            if sell.ts_parsed - buy.ts_parsed >= timedelta(days=366):
                return idx

        raise Exception("Something went wrong")

    def lowest_gains():
        # type: () -> int
        # This optimizes for paying least taxes now. Most open positions left for later.
        # If it's ST no matter what, take the smallest gain
        if sell.ts_parsed - buys[0].ts_parsed < timedelta(days=366):
            _, idx = max((b.cost / b.amt, idx) for idx, b in enumerate(buys))
            return idx
        # Otherwise take smallest gain that is LT
        _, idx = max((b.cost / b.amt, idx) for idx, b in enumerate(buys)
                     if sell.ts_parsed - b.ts_parsed >= timedelta(days=366))
        return idx

    # return 0  # FIFO: ST (23.7K + 576K), LT 597K
    # return len(buys) - 1  # LIFO: ST (23.6K + 575K), LT 597K
    return smart_lifo()  # ST: (23.7K + 424K), LT: 749K
    # return lowest_gains()  # ST: (23.5K + 424K), LT: 748.6K

def get_buys_sells_by_typ(common_rows_w_usd):
    # type: (List[CommonRowWUSD]) -> Dict[str, List[Union[Buy, Sell]]]
    buys = []
    sells = []
    for row in common_rows_w_usd:
        src_typ = row.pair[:3]
        dst_typ = row.pair[-3:]
        assert src_typ != 'USD'
        if dst_typ == 'USD':
            if row.src > 0:
                assert row.dst < 0
                buys.append(Buy(src_typ, row.src, -row.dst, row.site, row.ts_parsed))
            else:
                assert row.dst > 0
                sells.append(Sell(src_typ, -row.src, row.dst, row.site, row.ts_parsed))
        else:
            # Crypto-crypto exchanges
            # If we couldn't get the $ value of the currency, assume src/dst had same value
            assert row.src_usd is not None
            assert row.dst_usd is not None
            if row.src > 0:
                assert row.src_usd > 0
                assert row.dst < 0
                assert row.dst_usd < 0
                buys.append(Buy(src_typ, row.src, row.src_usd, row.site, row.ts_parsed))
                sells.append(Sell(dst_typ, -row.dst, -row.dst_usd, row.site, row.ts_parsed))
            else:
                assert row.src_usd < 0
                assert row.dst > 0
                assert row.dst_usd > 0
                buys.append(Buy(dst_typ, row.dst, row.dst_usd, row.site, row.ts_parsed))
                sells.append(Sell(src_typ, -row.src, -row.src_usd, row.site, row.ts_parsed))

    bs_by_typ = defaultdict(list)  # type: Dict[str, List[Union[Buy, Sell]]]
    for buy in buys:
        bs_by_typ[buy.typ].append(buy)
    for sell in sells:
        bs_by_typ[sell.typ].append(sell)
    # Sort by typ. Stable sorting ensures buys before sells
    for typ in bs_by_typ:
        bs_by_typ[typ] = sorted(bs_by_typ[typ], key=lambda row: (row.ts_parsed, type(row).__name__))
    return bs_by_typ

class GdaxPriceCache(object):
    def __init__(self):
        # type: () -> None
        self.gdax_price_cache = {}  # type: Dict[Tuple[str, str], Optional[float]]

    @staticmethod
    @contextmanager
    def deco():
        # type: () -> Iterator[GdaxPriceCache]
        gpc = GdaxPriceCache()
        with open('gdax_price_cache/gdax_price_cache.csv') as f:
            for row in csv.DictReader(f):
                ts = dateparse(row['ts']).isoformat()
                gpc.gdax_price_cache[(row['type'], ts)] = float(row['price']) if row['price'] else None

        yield gpc

        with open('gdax_price_cache/gdax_price_cache.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['type', 'ts', 'price'])
            for (typ, isots), result in gpc.gdax_price_cache.items():
                w.writerow([typ, isots, result])

    def get(self, typ, ts_parsed):
        # type: (str, datetime) -> Optional[float]
        isots = ts_parsed.isoformat()
        isots_end = (ts_parsed + timedelta(seconds=65)).isoformat()

        # Hard code ETC / DAO prices from the very early days.
        if typ in ('DAO'):
            if ts_parsed == datetime(2016, 7, 20, 15, 10, 51, tzinfo=tzutc()):
                return 0.11869335581813786

        if typ not in ('BTC', 'ETH', 'LTC'):
            # print("Gdax doesn't know about %s" % typ)
            return None

        if (typ, isots) not in self.gdax_price_cache:
            req = "{}/products/{}-USD/candles".format(GDAX_API, typ)
            params = {'granularity': "60", 'start': str(isots), 'end': str(isots_end)}
            result = requests.get(req, params).json()
            print("Req = {}. Params = {}\nResult = {}".format(req, params, result))
            if not result:
                print("No data from gdax")
                self.gdax_price_cache[(typ, isots)] = None
            else:
                print("Price of {} at {} was {}".format(typ, ts_parsed, result[0][1]))
                assert len(result) <= 3
                self.gdax_price_cache[(typ, isots)] = float(result[0][1])

            print("Sleeping 0.2 to backoff from gdax api")
            time.sleep(0.4)

        return self.gdax_price_cache[(typ, isots)]

def fetch_usd_equivalents(common_rows):
    # type: (List[CommonRow]) -> List[CommonRowWUSD]
    with GdaxPriceCache.deco() as gdax_price_cache:
        common_rows_w_usd = []
        for row in common_rows:
            if row.src <= 1e-6 and row.dst <= 1e-6:
                # Whatever. Irrelevant trade.
                continue

            src_usd = None
            dst_usd = None
            # Don't need to fetch USD trades
            if 'USD' not in row.pair:
                src_usd_rate = gdax_price_cache.get(row.pair[:3], row.ts_parsed)
                dst_usd_rate = gdax_price_cache.get(row.pair[-3:], row.ts_parsed)
                assert src_usd_rate or dst_usd_rate
                src_usd = src_usd_rate and row.src * src_usd_rate
                dst_usd = dst_usd_rate and row.dst * dst_usd_rate
                assert src_usd or dst_usd
                # If only one of the two rates were available, assume the trade was at
                # fair market value and src/dst amounts were worth the same in USD
                if src_usd is not None:
                    dst_usd = dst_usd or -src_usd
                if dst_usd is not None:
                    src_usd = src_usd or -dst_usd

            common_rows_w_usd.append(CommonRowWUSD(
                row.pair, row.src, row.dst, row.site, row.ts_parsed, src_usd, dst_usd,
            ))

        return common_rows_w_usd

def dedupe_bs(buys_sells):
    # type: (List[Union[Buy, Sell]]) -> List[Union[Buy, Sell]]
    """Dedupe things that happened on the same day"""
    by_day = defaultdict(list)  # type: Dict[Tuple[BuyOrSellType, str, str, datetime], List[Union[Buy, Sell]]]
    for row in buys_sells:
        by_day[(type(row), row.typ, row.site, datetime.combine(row.ts_parsed.date(), dt_time.min))].append(row)

    deduped_rows = []
    for (buy_or_sell, typ, site, ts_date), rows in by_day.items():
        assert len(rows) > 0
        if len(rows) == 1:
            row = rows[0]._replace(ts_parsed=ts_date)
            deduped_rows.append(row)
        elif buy_or_sell == Buy:
            assert all(isinstance(r, Buy) for r in rows)
            deduped_rows.append(Buy(
                typ,
                sum(r.amt for r in rows),
                sum(r.cost for r in rows),
                site + "[ {} rows combined ]".format(len(rows)),
                ts_date,
            ))
        elif buy_or_sell == Sell:
            assert all(isinstance(r, Sell) for r in rows)
            deduped_rows.append(Sell(
                typ,
                sum(r.amt for r in rows),
                sum(r.cost for r in rows),
                site + "[ {} rows combined ]".format(len(rows)),
                ts_date,
            ))

    # Sort by typ/date. Buys before sells
    deduped_rows = sorted(deduped_rows, key=lambda row: (row.ts_parsed, type(row).__name__))
    return deduped_rows

def dedupe_cb(cost_basis):
    # type: (List[CostBasis]) -> List[CostBasis]
    """Dedupe things that happened on the same day"""
    by_day = defaultdict(list)  # type: Dict[Tuple[str, datetime, datetime], List[CostBasis]]
    for row in cost_basis:
        by_day[(
            row.typ,
            datetime.combine(row.buy_ts.date(), dt_time.min),
            datetime.combine(row.sell_ts.date(), dt_time.min)
        )].append(row)

    deduped_rows = []
    for (typ, buy_ts, sell_ts), rows in by_day.items():
        assert len(rows) > 0
        if len(rows) == 1:
            row = rows[0]._replace(buy_ts=buy_ts, sell_ts=sell_ts)
            deduped_rows.append(row)
        else:
            deduped_rows.append(CostBasis(
                typ,
                sum(r.amt for r in rows),
                sum(r.proceeds for r in rows),
                sum(r.basis for r in rows),
                buy_ts,
                sell_ts,
                "[ {} rows combined ]".format(len(rows)),
                "[ {} rows combined ]".format(len(rows)),
            ))

    # Sort by typ/date. Buys before sells
    deduped_rows = sorted(deduped_rows, key=lambda row: (row.sell_ts, row.buy_ts))
    return deduped_rows

def convert_cex_row(row):
    # type: (Mapping[str, str]) -> Optional[CommonRow]
    if row['Type'] in ('deposit', 'withdraw'):
        # Just a transfer.
        return None
    elif row['Type'] in ('buy', 'sell') and row['Pair'] == '':
        # this is a goofy entry that doesn't correspond to an actual trade
        return None
    elif row['Type'] == 'sell' and row['Pair'] in ('ETH/USD', 'ETH/EUR'):
        comment = re.match("Sold (.*) ETH at (.*) (USD|EUR)", row['Comment'])
        assert comment is not None
        eth = -float(comment.group(1))
        price = float(comment.group(2))
        fiat = float(row['Amount'])
        assert abs(price * eth + float(row['FeeAmount']) + fiat) <= .01
        return CommonRow(row['Pair'], eth, fiat, 'CEX', row['DateUTC'],
                         dateparse(row['DateUTC']).replace(tzinfo=tzutc()))
    elif row['Type'] == 'buy' and row['Pair'] in ('ETH/USD', 'ETH/EUR'):
        comment = re.match("Bought (.*) ETH at (.*) USD", row['Comment'])
        assert comment is not None
        eth = float(comment.group(1))
        price = float(comment.group(2))
        assert eth == float(row['Amount'])
        fiat = -(price * eth)
        return CommonRow(row['Pair'], eth, fiat, 'CEX', row['DateUTC'],
                         dateparse(row['DateUTC']).replace(tzinfo=tzutc()))
    else:
        raise Exception("Unknown CEX row: %s" % row)

def test_convert_cex_row():
    # type: () -> None
    t = """DateUTC,Amount,Symbol,Balance,Type,Pair,FeeSymbol,FeeAmount,Comment
2017-12-20 09:55:03,795.59,USD,895.75,sell,ETH/USD,USD,1.98,Sold 0.88816258 ETH at 898 USD
2017-12-31 19:23:29,1.11126500,ETH,1.11126500,buy,ETH/USD,USD,2.29,Bought 1.11126500 ETH at 821.9999 USD"""
    f = StringIO(t)
    cex_rows = list(csv.DictReader(f))
    common_rows = convert_rows(cex_rows, convert_cex_row)
    assert common_rows == [
        CommonRow("ETH/USD", -0.88816258, 795.59, "CEX", "2017-12-20 09:55:03",
                  datetime(2017, 12, 20, 9, 55, 3, tzinfo=tzutc())),
        CommonRow("ETH/USD", 1.11126500, -(1.11126500 * 821.9999), "CEX", "2017-12-31 19:23:29",
                  datetime(2017, 12, 31, 19, 23, 29, tzinfo=tzutc())),
    ]

def convert_coinbase_row(row):
    # type: (Mapping[str, str]) -> Optional[CommonRow]
    crypto_type = row['Currency']
    fiat_type = row['Transfer Total Currency']
    if not (crypto_type and fiat_type):
        # Just a plain transfer - no exchange
        return None

    pair = crypto_type + '/' + fiat_type
    crypto = float(row['Amount'])
    fiat = float(row['Transfer Total'])
    assert fiat > 0 and crypto != 0
    if crypto > 0:
        fiat = -fiat
    return CommonRow(pair, crypto, fiat, 'Coinbase', row['Timestamp'], dateparse(row['Timestamp']))

def test_convert_coinbase_row():
    # type: () -> None
    t = """Timestamp,Balance,Amount,Currency,To,Notes,Instantly Exchanged,Transfer Total,Transfer Total Currency,Transfer Fee,Transfer Fee Currency,Transfer Payment Method,Transfer ID,Order Price,Order Currency,Order BTC,Order Tracking Code,Order Custom Parameter,Order Paid Out,Recurring Payment ID,Coinbase ID (visit https://www.coinbase.com/transactions/[ID] in your browser),Bitcoin Hash (visit https://www.coinbase.com/tx/[HASH] in your browser for more info)
2016-04-03 12:44:17 -0700,0.04714286,0.04714286,BTC,570170d6ede0d822170003cb,Bought 0.04714286 BTC for $20.00 USD.,false,20.0,USD,0.2,USD,Bank of America - BofA... ********9496,5701728b2ee3cd04b500048b,"","","","","","","",57017291e42ae13d2a00020b,""
2017-07-11 17:04:58 -0700,8.28632484,-1.0,ETH,572a417ae1764802e1000599,596567a9ea216a0222536a1d,false,187.53,USD,2.99,USD,USD Wallet,596567a9ea216a0222536a1d,"","","","","","","",596567aa2122dd0001f25f0d,""
"""
    f = StringIO(t)
    coinbase_rows = list(csv.DictReader(f))
    common_rows = convert_rows(coinbase_rows, convert_coinbase_row)
    assert common_rows == [
        CommonRow("BTC/USD", 0.04714286, -20.0, "Coinbase", "2016-04-03 12:44:17 -0700",
                  datetime(2016, 4, 3, 12, 44, 17, tzinfo=tzoffset(None, -25200))),
        CommonRow("ETH/USD", -1.0, 187.53, "Coinbase", "2017-07-11 17:04:58 -0700",
                  datetime(2017, 7, 11, 17, 4, 58, tzinfo=tzoffset(None, -25200))),
    ]

kraken_cache = {}  # type: Dict[str, Mapping[str, str]]
def convert_kraken_row(row):
    # type: (Mapping[str, str]) -> Optional[CommonRow]
    if row['type'] in ('deposit', 'withdrawal', 'transfer'):
        return None
    if row['type'] != 'trade':
        raise Exception("Unknown kraken row: %s" % row)
    if row['refid'] not in kraken_cache:
        # First half of pair
        kraken_cache[row['refid']] = row
        return None

    asset_map = {
        'ZUSD': 'USD',
        'XETC': 'ETC',
        'XETH': 'ETH',
        'XXBT': 'BTC',
    }
    prev_row = kraken_cache.pop(row['refid'])
    prev_amt = float(prev_row['amount'])
    amt = float(row['amount'])
    prev_asset = asset_map[prev_row['asset']]
    asset = asset_map[row['asset']]
    pair = prev_asset + '/' + asset

    assert (prev_amt < 0 and amt > 0 or prev_amt > 0 and amt < 0)
    assert prev_row['time'] == row['time']

    return CommonRow(pair, prev_amt, amt, 'Kraken', row['time'], dateparse(row['time']).replace(tzinfo=tzutc()))

def test_convert_kraken_row():
    # type: () -> None
    t = """"txid","refid","time","type","aclass","asset","amount","fee","balance"
"LQT3EO-D4PO7-6QCRZP","TZBXCO-6DHLN-ZO22HQ","2016-04-23 21:23:03","trade","currency","XETH",-2.0000000000,0.0000000000,8.0000000000
"LRY4II-TYZCS-QLC4FT","TZBXCO-6DHLN-ZO22HQ","2016-04-23 21:23:03","trade","currency","ZUSD",16.9710,0.0441,16.9269
"LUJMM6-BTILY-SH6QFA","TXDUA7-QGSEW-5AHH7Q","2016-06-17 11:32:53","trade","currency","XETH",-8.0410719900,0.0000000000,1591.9589378300
"LX6PV7-CN5YE-Z4HNT7","TXDUA7-QGSEW-5AHH7Q","2016-06-17 11:32:53","trade","currency","XXBT",0.1861590000,0.0004840000,0.1856750000
"""
    f = StringIO(t)
    kraken_rows = list(csv.DictReader(f))
    common_rows = convert_rows(kraken_rows, convert_kraken_row)
    assert common_rows == [
        CommonRow("ETH/USD", -2.0, 16.9710, "Kraken", "2016-04-23 21:23:03",
                  datetime(2016, 4, 23, 21, 23, 3, tzinfo=tzutc())),
        CommonRow("ETH/BTC", -8.0410719900, 0.1861590000, "Kraken", "2016-06-17 11:32:53",
                  datetime(2016, 6, 17, 11, 32, 53, tzinfo=tzutc())),
    ]

def convert_poloniex_row(row):
    # type: (Mapping[str, str]) -> CommonRow
    assert row['Category'] == 'Exchange'
    if row['Type'] not in ('Buy', 'Sell'):
        raise Exception("Unknown poloniex row: %s" % row)

    src = float(row['Amount'])
    dst = float(row['Total'])
    assert src >= 0 and dst >= 0
    if row['Type'] == 'Buy':
        dst = -dst
    if row['Type'] == 'Sell':
        src = -src
    return CommonRow(row['Market'], src, dst, 'Poloniex', row['Date'], dateparse(row['Date']).replace(tzinfo=tzutc()))

def test_convert_poloniex_row():
    # type: () -> None
    t = """Date,Market,Category,Type,Price,Amount,Total,Fee,Order Number,Base Total Less Fee,Quote Total Less Fee
2016-06-29 03:54:56,XEM/BTC,Exchange,Buy,0.00001934,27774.58651931,0.53716050,0.15%,2311153645,-0.53716050,27732.92463954
2017-12-16 23:41:09,XEM/BTC,Exchange,Sell,0.00003310,6429.67600000,0.21282227,0.25%,38237473363,0.21229022,-6429.67600000
"""
    f = StringIO(t)
    poloniex_rows = list(csv.DictReader(f))
    common_rows = convert_rows(poloniex_rows, convert_poloniex_row)
    assert common_rows == [
        CommonRow("XEM/BTC", 27774.58651931, -0.53716050, "Poloniex", "2016-06-29 03:54:56",
                  datetime(2016, 6, 29, 3, 54, 56, tzinfo=tzutc())),
        CommonRow("XEM/BTC", -6429.67600000, 0.21282227, "Poloniex", "2017-12-16 23:41:09",
                  datetime(2017, 12, 16, 23, 41, 9, tzinfo=tzutc())),
    ]
def convert_gdax_rows(rows):
    # type: (Sequence[Dict[str, str]]) -> List[CommonRow]
    gdax_trades = defaultdict(list)  # type: Dict[str, List[Dict[str, str]]]
    for row in rows:
        if row['trade id']:
            gdax_trades[row['trade id']].append(row)

    units_to_pair = {
        frozenset(['ETH', 'USD']): ('ETH/USD', 'ETH', 'USD'),
        frozenset(['ETH', 'BTC']): ('BTC/ETH', 'BTC', 'ETH'),
    }

    common_rows = []
    for trade_id, trade in gdax_trades.items():
        # Each trade should happen at one timestamp
        tses = set([trade_item['time'] for trade_item in trade])
        assert len(tses) == 1, "tses={}".format(tses)
        ts = tses.pop()

        # Two matches, and one fee
        assert len(trade) == 3
        unit_to_amount = defaultdict(float)  # type: Dict[str, float]
        for trade_item in trade:
            assert trade_item['type'] in ('match', 'fee')
            unit_to_amount[trade_item['amount/balance unit']] += float(trade_item['amount'])

        pair, left, right = units_to_pair[frozenset(unit_to_amount)]
        assert left in unit_to_amount
        assert right in unit_to_amount
        common_rows.append(CommonRow(
            pair, unit_to_amount[left], unit_to_amount[right], 'GDAX', ts, dateparse(ts)))

    return common_rows

def convert_rows(rows, convert_func):
    # type: (Sequence[Mapping[str, str]], Callable[[Mapping[str, str]], Optional[CommonRow]]) -> List[CommonRow]
    common_rows = []
    for row in rows:
        conv = convert_func(row)
        if conv:
            common_rows.append(conv)
    assert not kraken_cache
    return common_rows

def import_2016_2017():
    # type: () -> List[CommonRow]
    with open('csvs/123117-CEX.csv') as f:
        cex_rows = list(csv.DictReader(f))

    with open('csvs/123117-Coinbase-BTC.csv') as f:
        for _ in range(4):
            f.readline()
        cb_btc_rows = list(csv.DictReader(f))

    with open('csvs/123117-Coinbase-ETH.csv') as f:
        for _ in range(4):
            f.readline()
        cb_eth_rows = list(csv.DictReader(f))

    with open('csvs/123117-GDAX-BTC.csv') as f:
        gdax_btc_rows = list(csv.DictReader(f))

    with open('csvs/123117-GDAX-ETH.csv') as f:
        gdax_eth_rows = list(csv.DictReader(f))

    with open('csvs/123117-GDAX-USD.csv') as f:
        gdax_usd_rows = list(csv.DictReader(f))

    with open('csvs/123117-Kraken.csv') as f:
        kraken_rows = list(csv.DictReader(f))

    with open('csvs/123117-Poloniex-TradeHistory.csv') as f:
        poloniex_rows = list(csv.DictReader(f))

    print("Orig row Counts")
    pprint([len(cex_rows), len(cb_btc_rows), len(cb_eth_rows),
            len(gdax_btc_rows), len(gdax_eth_rows), len(gdax_usd_rows),
            len(kraken_rows), len(poloniex_rows)])

    common_cex_rows = convert_rows(cex_rows, convert_cex_row)
    common_cb_btc_rows = convert_rows(cb_btc_rows, convert_coinbase_row)
    common_cb_eth_rows = convert_rows(cb_eth_rows, convert_coinbase_row)
    common_kraken_rows = convert_rows(kraken_rows, convert_kraken_row)
    common_poloniex_rows = convert_rows(poloniex_rows, convert_poloniex_row)
    common_gdax_rows = convert_gdax_rows(gdax_btc_rows + gdax_eth_rows + gdax_usd_rows)

    common_extra_rows = [
        CommonRow("BTC/ETH", -5.682, 264.667, 'HitBTC????', "04/17/2016", datetime(2016, 4, 17, 4, 15, 41,
                  tzinfo=tzutc())),
        CommonRow("ETC/USD", 3839.75349869, -0.01, 'DAO Split', "06/17/2016", datetime(2016, 7, 20, 1, 20, 40,
                  tzinfo=tzutc())),
    ]

    print("Common row counts")
    pprint([len(common_cex_rows), len(common_cb_btc_rows), len(common_cb_eth_rows),
            len(common_gdax_rows),
            len(common_kraken_rows), len(common_poloniex_rows)])

    print("Example common rows:")
    pprint([common_cex_rows[0], common_cb_btc_rows[0], common_cb_eth_rows[0],
            common_gdax_rows[0],
            common_kraken_rows[0], common_poloniex_rows[0]])

    common_rows = sorted(
        common_cex_rows + common_cb_btc_rows + common_cb_eth_rows +
        common_kraken_rows + common_poloniex_rows + common_gdax_rows + common_extra_rows,
        key=lambda row: row.ts_parsed,
    )
    return common_rows

def import_2018():
    # type: () -> List[CommonRow]
    raise NotImplementedError()

def make_intermediate(common_rows):
    # type: (List[CommonRow]) -> Dict[str, List[CostBasis]]

    print("Total common rows:")
    pprint(len(common_rows))

    if os.path.exists('intermediate'):
        shutil.rmtree('intermediate')
    os.mkdir('intermediate')

    with open('intermediate/01-123117-rows_interleaved.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(CommonRow._fields)
        w.writerows(common_rows)

    common_rows_w_usd = fetch_usd_equivalents(common_rows)

    with open('intermediate/02-123117-rows_interleaved_w_usd.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(CommonRowWUSD._fields)
        w.writerows(common_rows_w_usd)

    bs_by_typ = get_buys_sells_by_typ(common_rows_w_usd)

    # Dedupe buys_sells to write to file for observation
    deduped_bs_by_typ = {typ: dedupe_bs(buys_sells) for typ, buys_sells in bs_by_typ.items()}
    with open('intermediate/03-123117-rows_deduped_by_day.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['BuyOrSell'] + list(Buy._fields))
        for typ, bses in deduped_bs_by_typ.items():
            w.writerows([[type(bs).__name__] + [str(field) for field in bs] for bs in bses])

    # FINALLY. Get cost basis and remaining assets.
    cb_by_typ = {typ: cost_basis(buys_sells) for typ, buys_sells in bs_by_typ.items()}

    with open('intermediate/04-123117-cost_basis.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(CostBasis._fields)
        for typ, (cbs, assets) in cb_by_typ.items():
            w.writerows(cbs)

    with open('intermediate/05-123117-remaining_assets.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(Buy._fields)
        for typ, (cbs, assets) in cb_by_typ.items():
            w.writerows(assets)

    # Dedupe cost basis by day to make filing easier
    deduped_cb_by_typ = {typ: dedupe_cb(cost_basis) for typ, (cost_basis, leftover) in cb_by_typ.items()}
    pprint(deduped_cb_by_typ)

    with open('intermediate/06-123117-cb_deduped_by_day.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(list(CostBasis._fields))
        for typ, cbs in deduped_cb_by_typ.items():
            w.writerows(cbs)

    for yr in (2016, 2017, 2018):
        with open('intermediate/07-%d-cb_for_easytxf.csv' % yr, 'w') as f:
            w = csv.writer(f)
            w.writerow(["Symbol", "Quantity", "Date Acquired", "Date Sold", "Proceeds",
                        "Cost Basis", "Gain (or loss)", "Sale Category"])
            for typ, cbs in deduped_cb_by_typ.items():
                w.writerows([
                    [
                        cb.typ, cb.amt, cb.buy_ts.isoformat(), cb.sell_ts.isoformat(),
                        cb.proceeds, cb.basis, cb.proceeds - cb.basis,
                        "Box C [Short term unreported]" if cb.sell_ts - cb.buy_ts < timedelta(days=366)
                        else "Box F [Long term unreported]",
                    ]
                    for cb in cbs
                    if cb.sell_ts.year == yr
                ])

    print("Total deduped rows:")
    pprint({typ: len(bs) for typ, bs in deduped_cb_by_typ.items()})

    return deduped_cb_by_typ

def printout(deduped_cb_by_typ):
    # type: (Dict[str, List[CostBasis]]) -> None
    totals = {
        2016: {},
        2017: {},
    }  # type: Dict[int, Dict[str, float]]
    for yr in (2016, 2017):
        totals[yr]['st_proceeds'] = sum(
            r.proceeds for cbs in deduped_cb_by_typ.values() for r in cbs
            if r.sell_ts - r.buy_ts < timedelta(days=366) and r.sell_ts.year == yr
        )
        totals[yr]['st_basis'] = sum(
            r.basis for cbs in deduped_cb_by_typ.values() for r in cbs
            if r.sell_ts - r.buy_ts < timedelta(days=366) and r.sell_ts.year == yr
        )
        totals[yr]['lt_proceeds'] = sum(
            r.proceeds for cbs in deduped_cb_by_typ.values() for r in cbs
            if r.sell_ts - r.buy_ts >= timedelta(days=366) and r.sell_ts.year == yr
        )
        totals[yr]['lt_basis'] = sum(
            r.basis for cbs in deduped_cb_by_typ.values() for r in cbs
            if r.sell_ts - r.buy_ts >= timedelta(days=366) and r.sell_ts.year == yr
        )

        totals[yr]['st_gain'] = totals[yr]['st_proceeds'] - totals[yr]['st_basis']
        totals[yr]['lt_gain'] = totals[yr]['lt_proceeds'] - totals[yr]['lt_basis']
    pprint(totals)

def run_2016_2017():
    # type: () -> None
    common_rows = import_2016_2017()
    deduped_cb_by_typ = make_intermediate(common_rows)
    printout(deduped_cb_by_typ)

    # import IPython; IPython.embed()

def run_2018():
    # type: () -> None
    common_rows = import_2018()
    import IPython; IPython.embed()

if __name__ == "__main__":
    run_2016_2017()
