from __future__ import print_function

import csv
import re

from collections import (
    namedtuple,
)
from pprint import pprint

CommonRow = namedtuple('CommonRow', ['pair', 'src', 'dst', 'ts'])

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
        return CommonRow(row['Pair'], eth, fiat, row['DateUTC'])
    elif row['Type'] == 'buy' and row['Pair'] in ('ETH/USD', 'ETH/EUR'):
        comment = re.match("Bought (.*) ETH at (.*) USD", row['Comment'])
        eth, price = comment.group(1, 2)
        eth = float(eth)
        price = float(price)
        assert eth == float(row['Amount'])
        fiat = -(price * eth)
        return CommonRow(row['Pair'], eth, fiat, row['DateUTC'])
    else:
        raise Exception("Unknown CEX row: %s" % row)

def convert_coinbase_row(cb_row):
    # type: (Dict[str, str]) -> CommonRow
    pass

def convert_rows(rows, convert_func):
    # type: (List[Dict[str, str]], Callable[[Dict[str, str]], CommonRow]) -> List[CommonRow]
    common_rows = []
    for row in rows:
        conv = convert_func(row)
        if conv:
            common_rows.append(conv)
    return common_rows

if __name__ == "__main__":
    with open('csvs/123117-CEX.csv') as f:
        cex_rows = list(csv.DictReader(f))

    with open('csvs/123117-Coinbase-BTC.csv') as f:
        for _ in range(4):
            f.readline()
        cb_btc_rows = list(csv.DictReader(f))

    with open('csvs/123117-Coinbase-ETH.csv') as f:
        cb_eth_rows = list(csv.DictReader(f))

    with open('csvs/123117-Kraken.csv') as f:
        kraken_rows = list(csv.DictReader(f))

    with open('csvs/123117-Poloniex-TradeHistory.csv') as f:
        poloniex_rows = list(csv.DictReader(f))

    pprint([len(cex_rows), len(cb_btc_rows), len(cb_eth_rows), len(kraken_rows), len(poloniex_rows)])

    common_rows = convert_rows(cex_rows, convert_cex_row)

    pprint(cb_btc_rows[:2])
