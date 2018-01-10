from __future__ import print_function

import csv
import re

from collections import (
    namedtuple,
)
from pprint import pprint

CommonRow = namedtuple('CommonRow', ['pair', 'src', 'dst', 'site', 'ts'])

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
        return CommonRow(row['Pair'], eth, fiat, 'CEX', row['DateUTC'])
    elif row['Type'] == 'buy' and row['Pair'] in ('ETH/USD', 'ETH/EUR'):
        comment = re.match("Bought (.*) ETH at (.*) USD", row['Comment'])
        eth, price = comment.group(1, 2)
        eth = float(eth)
        price = float(price)
        assert eth == float(row['Amount'])
        fiat = -(price * eth)
        return CommonRow(row['Pair'], eth, fiat, 'CEX', row['DateUTC'])
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
    return CommonRow(pair, crypto, fiat, 'Coinbase', row['Timestamp'])

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

    return CommonRow(pair, prev_amt, amt, 'Kraken', row['time'])

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
        src = -dst
    return CommonRow(row['Market'], src, dst, 'Poloniex', row['Date'])

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

    with open('csvs/123117-Kraken.csv') as f:
        kraken_rows = list(csv.DictReader(f))

    with open('csvs/123117-Poloniex-TradeHistory.csv') as f:
        poloniex_rows = list(csv.DictReader(f))

    pprint([len(cex_rows), len(cb_btc_rows), len(cb_eth_rows), len(kraken_rows), len(poloniex_rows)])

    common_cex_rows = convert_rows(cex_rows, convert_cex_row)
    common_cb_btc_rows = convert_rows(cb_btc_rows, convert_coinbase_row)
    common_cb_eth_rows = convert_rows(cb_eth_rows, convert_coinbase_row)
    common_kraken_rows = convert_rows(kraken_rows, convert_kraken_row)
    common_poloniex_rows = convert_rows(poloniex_rows, convert_poloniex_row)
