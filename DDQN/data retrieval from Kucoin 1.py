import requests
import hmac
import hashlib
import base64
import time
import csv

api_key = ""
api_secret = ""
api_passphrase = ""

symbol = 'VET'
interval = '30min'  # Add desired intervals to this list

now_prices = int((time.time()) * 1000)


start_times = [
    1543795200,
    1544611200,
    1545427200,
    1546243200,
    1547059200,
    1547875200,
    1548691200,
    1549507200,
    1550323200,
    1551139200,
    1551955200,
    1552771200,
    1553587200,
    1554403200,
    1555219200,
    1556035200,
    1556851200,
    1557667200,
    1558483200,
    1559299200,
    1560115200,
    1560931200,
    1561747200,
    1562563200,
    1563379200,
    1564195200,
    1565011200,
    1565827200,
    1566643200,
    1567459200,
    1568275200,
    1569091200,
    1569907200,
    1570723200,
    1571539200,
    1572355200,
    1573171200,
    1573987200,
    1574803200,
    1575619200,
    1576435200,
    1577251200,
    1578067200,
    1578883200,
    1579699200,
    1580515200,
    1581331200,
    1582147200,
    1582963200,
    1583779200,
    1584595200,
    1585411200,
    1586227200,
    1587043200,
    1587859200,
    1588675200,
    1589491200,
    1590307200,
    1591123200,
    1591939200,
    1592755200,
    1593571200,
    1594387200,
    1595203200,
    1596019200,
    1596835200,
    1597651200,
    1598467200,
    1599283200,
    1600099200,
    1600915200,
    1601731200,
    1602547200,
    1603363200,
    1604179200,
    1604995200,
    1605811200,
    1606627200,
    1607443200,
    1608259200,
    1609075200,
    1609891200,
    1610707200,
    1611523200,
    1612339200,
    1613155200,
    1613971200,
    1614787200,
    1615603200,
    1616419200,
    1617235200,
    1618051200,
    1618867200,
    1619683200,
    1620499200,
    1621315200,
    1622131200,
    1622947200,
    1623763200,
    1624579200,
    1625395200,
    1626211200,
    1627027200,
    1627027200,
    1627843200,
    1628659200,
    1629475200,
    1630291200,
    1631107200,
    1631923200,
    1632739200,
    1633555200,
    1634371200,
    1635187200,
    1636003200,
    1636819200,
    1637635200,
    1638451200,
    1639267200,
    1640083200,
    1640899200,
    1641715200,
    1642531200,
    1643347200,
    1644163200,
    1644979200,
    1645795200,
    1646611200,
    1647427200,
    1648243200,
    1649059200,
    1649875200,
    1650691200,
    1651507200,
    1652323200,
    1653139200,
    1653955200,
    1654771200,
    1655587200,
    1656403200,
    1657219200,
    1658035200,
    1658851200,
    1659667200,
    1660483200,
    1661299200,
    1662115200,
    1662931200,
    1663747200,
    1664563200,
    1665379200,
    1666195200,
    1667011200,
    1667827200,
    1668643200,
    1669459200,
    1670275200,
    1671091200,
    1671907200,
    1672723200,
    1673539200,
    1674355200,
    1675171200,
    1675987200,
    1676803200,
    1677619200,
    1678435200,
    1679251200,
    1680067200,
    1680883200,
    1681336800
]

csv_file = 'klines_vet_data.csv'

# Create the CSV file with headers if it doesn't exist
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["time", "open", "close", "high", "low", "volume", "turnover"])


for i in range(len(start_times) - 1):
    url_klines = f'https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol}-USDT&startAt={start_times[i]}&endAt={start_times[i+1]}'
    str_to_sign_klines = str(now_prices) + f'GET/api/v1/market/candles?type={interval}&symbol={symbol}-USDT&startAt={start_times[i]}&endAt={start_times[i+1]}'
    signature_klines = base64.b64encode(hmac.new(api_secret.encode('utf-8'), str_to_sign_klines.encode('utf-8'), hashlib.sha256).digest())
    passphrase_klines = base64.b64encode(hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
    headers_klines = {
        "KC-API-SIGN": signature_klines,
        "KC-API-TIMESTAMP": str(now_prices),
        "KC-API-KEY": api_key,
        "KC-API-PASSPHRASE": passphrase_klines,
        "KC-API-KEY-VERSION": "2"
    }
    response_klines = requests.request('get', url_klines, headers=headers_klines)

    if response_klines.status_code == 200:
        print(f"{interval} Klines data {i}: {response_klines.json()}")
        data = response_klines.json()['data'] # extract 'data' field from JSON response
        data.reverse()  # reverse the order of the data list
        # Append the data to the CSV file
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for row in data:
                writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])

    else:
        print(f"Error fetching {interval} data {i}: {response_klines.status_code}, {response_klines.text}")
