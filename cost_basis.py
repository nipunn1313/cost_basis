from __future__ import (
    division,
    print_function,
)

import csv
import re
import requests
import time

from collections import (
    defaultdict,
    namedtuple,
)
from contextlib import contextmanager
from datetime import datetime, timedelta
from dateutil.parser import parse as dateparse
from dateutil.tz import tzutc
from pprint import pprint

GDAX_API = 'https://api.gdax.com'

CommonRow = namedtuple('CommonRow', ['pair', 'src', 'dst', 'site', 'ts_orig', 'ts_parsed'])
CommonRowWUSD = namedtuple('CommonRowWUSD', ['pair', 'src', 'dst', 'site', 'ts_parsed', 'src_usd', 'dst_usd'])
Buy = namedtuple('Buy', ['typ', 'amt', 'cost', 'site', 'ts_parsed'])
Sell = namedtuple('Sell', ['typ', 'amt', 'cost', 'site', 'ts_parsed'])
CostBasis = namedtuple('CostBasis', ['typ', 'amt', 'cost', 'basis', 'buy_ts', 'sell_ts'])

def cost_basis(buys_sells):
    # type: (List[Union[Buy, Sell]]) -> Tuple[List[CostBasis], List[Buy]]
    """Returns cost basis of realized gains + leftover buys"""
    cbs = []
    buys = []
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
                cbs.append(CostBasis(sell.typ, sell.amt, sell.cost, basis, buy.ts_parsed, sell.ts_parsed))

                if buys[buy_idx].cost <= 1e-6:
                    del buys[buy_idx]
                sell = sell._replace(amt=0, cost=0)
            else:
                # Split the sell
                assert buy.amt <= sell.amt
                partial_sell_cost = sell.cost * (buy.amt / sell.amt)
                cbs.append(CostBasis(sell.typ, buy.amt, partial_sell_cost, buy.cost, buy.ts_parsed, sell.ts_parsed))

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
    # return 0  # FIFO: ST 600K, LT 597K
    # return len(buys) - 1  # LIFO: ST 463K, LT 732K

    # If it's ST no matter what, take the shortest
    # This gets ST: 449K, LT: 747K
    if sell.ts_parsed - buys[0].ts_parsed < timedelta(days=366):
        return len(buys) - 1
    # Otherwise take the first LT
    for idx, buy in reversed(list(enumerate(buys))):
        if sell.ts_parsed - buy.ts_parsed >= timedelta(days=366):
            return idx
    raise Exception("Something went wrong")

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

    bs_by_typ = defaultdict(list)
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
        self.gdax_price_cache = {}

    @staticmethod
    @contextmanager
    def deco():
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
            params = {'granularity': 60, 'start': isots, 'end': isots_end}
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
    # type: List[CommonRow] -> List[CommonRowWUSD]
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
                src_usd = src_usd or -dst_usd
                dst_usd = dst_usd or -src_usd

            common_rows_w_usd.append(CommonRowWUSD(
                row.pair, row.src, row.dst, row.site, row.ts_parsed, src_usd, dst_usd,
            ))

        return common_rows_w_usd

def dedupe(buys_sells):
    # type: List[Union[Buys, Sells]] -> List[Union[Buys, Sells]]
    """Dedupe things that happened on the same day"""
    by_day = defaultdict(list)
    for row in buys_sells:
        by_day[(type(row), row.typ, row.site, row.ts_parsed.date())].append(row)

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

def convert_cex_row(row):
    # type: (Dict[str, str]) -> CommonRow
    if row['Type'] in ('deposit', 'withdraw'):
        # Just a transfer.
        return None
    elif row['Type'] in ('buy', 'sell') and row['Pair'] == '':
        # this is a goofy entry that doesn't correspond to an actual trade
        return None
    elif row['Type'] == 'sell' and row['Pair'] in ('ETH/USD', 'ETH/EUR'):
        comment = re.match("Sold (.*) ETH at (.*) (USD|EUR)", row['Comment'])
        eth, price = comment.group(1, 2)
        eth = -float(eth)
        price = float(price)
        fiat = float(row['Amount'])
        assert abs(price * eth + float(row['FeeAmount']) + fiat) <= .01
        return CommonRow(row['Pair'], eth, fiat, 'CEX', row['DateUTC'],
                         dateparse(row['DateUTC']).replace(tzinfo=tzutc()))
    elif row['Type'] == 'buy' and row['Pair'] in ('ETH/USD', 'ETH/EUR'):
        comment = re.match("Bought (.*) ETH at (.*) USD", row['Comment'])
        eth, price = comment.group(1, 2)
        eth = float(eth)
        price = float(price)
        assert eth == float(row['Amount'])
        fiat = -(price * eth)
        return CommonRow(row['Pair'], eth, fiat, 'CEX', row['DateUTC'],
                         dateparse(row['DateUTC']).replace(tzinfo=tzutc()))
    else:
        raise Exception("Unknown CEX row: %s" % row)

def convert_coinbase_row(row):
    # type: (Dict[str, str]) -> CommonRow
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

kraken_cache = {}  # type: Dict[str, Tuple[str, str]]
def convert_kraken_row(row):
    # type: (Dict[str, str]) -> CommonRow
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

def convert_poloniex_row(row):
    # type: (Dict[str, str]) -> CommonRow
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

def convert_gdax_rows(rows):
    # type: (List[Dict[str, str]]) -> List[CommonRow]
    gdax_trades = defaultdict(list)
    for row in gdax_btc_rows + gdax_eth_rows + gdax_usd_rows:
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
        unit_to_amount = defaultdict(float)
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
    # type: (List[Dict[str, str]], Callable[[Dict[str, str]], CommonRow]) -> List[CommonRow]
    common_rows = []
    for row in rows:
        conv = convert_func(row)
        if conv:
            common_rows.append(conv)
    assert not kraken_cache
    return common_rows

if __name__ == "__main__":
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

    print("Total common rows:")
    pprint(len(common_rows))

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

    deduped_bs_by_typ = {typ: dedupe(buys_sells) for typ, buys_sells in bs_by_typ.items()}

    with open('intermediate/03-123117-rows_deduped_by_day.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['BuyOrSell'] + list(Buy._fields))
        for typ, bses in deduped_bs_by_typ.items():
            w.writerows([[type(bs).__name__] + list(bs) for bs in bses])

    print("Total deduped rows:")
    pprint({typ: len(bs) for typ, bs in deduped_bs_by_typ.items()})

    # FINALLY. Get cost basis
    cb_by_typ = {typ: cost_basis(buys_sells) for typ, buys_sells in deduped_bs_by_typ.items()}

    pprint(cb_by_typ)

    short_term_16 = sum(r.cost - r.basis for cbs in cb_by_typ.values() for r in cbs[0]
                        if r.sell_ts - r.buy_ts < timedelta(days=366) and r.sell_ts.year == 2016)
    short_term_17 = sum(r.cost - r.basis for cbs in cb_by_typ.values() for r in cbs[0]
                        if r.sell_ts - r.buy_ts < timedelta(days=366) and r.sell_ts.year == 2017)
    short_term_18 = sum(r.cost - r.basis for cbs in cb_by_typ.values() for r in cbs[0]
                        if r.sell_ts - r.buy_ts < timedelta(days=366) and r.sell_ts.year == 2018)
    long_term_16 = sum(r.cost - r.basis for cbs in cb_by_typ.values() for r in cbs[0]
                       if r.sell_ts - r.buy_ts >= timedelta(days=366) and r.sell_ts.year == 2016)
    long_term_17 = sum(r.cost - r.basis for cbs in cb_by_typ.values() for r in cbs[0]
                       if r.sell_ts - r.buy_ts >= timedelta(days=366) and r.sell_ts.year == 2017)
    long_term_18 = sum(r.cost - r.basis for cbs in cb_by_typ.values() for r in cbs[0]
                       if r.sell_ts - r.buy_ts >= timedelta(days=366) and r.sell_ts.year == 2018)
    pprint(short_term_16)
    pprint(short_term_17)
    pprint(short_term_18)
    pprint(long_term_16)
    pprint(long_term_17)
    pprint(long_term_18)
