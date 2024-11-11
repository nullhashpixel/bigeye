import json
import ssl
import threading
from websockets.sync.client import connect as ws_connect

class OgmiosConnection:
    def __init__(self, ws_url, config=None):
        self.ws_url = ws_url
        self.config = config

        if ".demeter.run" in ws_url:
            if self.config.get('OGMIOS_SHARED_CONNECTION'):
                print("OGMIOS: using shared connection to a demeter.run instance.")
            else:
                print("\x1b[31mOGMIOS: using demeter.run instance but no shared connection. This might exceed free tier limits. Set OGMIOS_SHARED_CONNECTION=true to use a shared connection.\x1b[0m")
        else:
            if self.config.get('OGMIOS_SHARED_CONNECTION'):
                print("OGMIOS: using shared connection to a non-demeter instance. This might lead to slightly worse performance.")
            else:
                print("OGMIOS: using Ogmios")

        self.ws_lock = threading.Lock()

        self.ws = None
        self.connect()

        self.queries = {
                'NextBlock':         OgmiosQuery({"method": "nextBlock"}, self),
                'FindIntersect':     OgmiosQuery({"method": "findIntersection"}, self),
                'LocalStateAcquire': OgmiosQuery({"method": "acquireLedgerState"}, self),
                'LocalStateQuery':   OgmiosQuery({"method": "queryLedgerState/utxo"}, self),
                'LocalStateRelease': OgmiosQuery({"method": "releaseLedgerState"}, self),
                'QueryNetworkTip':   OgmiosQuery({"method": "queryNetwork/tip"}, self),
                'SubmitTransaction': OgmiosQuery({"method": "submitTransaction"}, self),
                'EvaluateTransaction': OgmiosQuery({"method": "evaluateTransaction"}, self),
                'AcquireMempool': OgmiosQuery({"method": "acquireMempool"}, self),
                'NextTransaction': OgmiosQuery({"method": "nextTransaction"}, self),
                'ProtocolParameters': OgmiosQuery({"method": "queryLedgerState/protocolParameters"}, self),
                }

    def connect(self):
        try:
            self.do_connect()
            return True
        except Exception as e:
            print("WS error:", e)

        print("trying again...")
        self.do_connect()
        return True

    def disconnect(self):
        try:
            self.ws.close()
            return True
        except Exception as e:
            print(e)

        return False

    def do_connect(self):
        ctx = None
        if self.config is not None and self.config.get('OGMIOS_DISABLE_CERTIFICATE_CHECK'):
            if self.ws_url.startswith('ws:/') or self.ws_url.startswith('http:/'):
                print("OGMIOS: WARNING: OGMIOS_DISABLE_CERTIFICATE_CHECK is only valid for SSL (wss) connections, => ignored.")
            else:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
        self.ws = ws_connect(self.ws_url, ssl_context=ctx)

    def send_recv(self, req):
        with self.ws_lock:
            success = False
            try:
                self.ws.send(req)
                success = True
            except Exception as e:
                print("WS error:", e)

            if not success:
                print("reconnecting...")
                self.connect()
                self.ws.send(req)

            try:
                m = self.ws.recv()
                return m
            except Exception as e:
                print("WS error:", e)
            print("retry recv...")
            
            m = self.ws.recv()
            return m

    def get_url(self):
        return self.ws_url

    def query(self, query_name, params={}):
        return self.queries[query_name].run(params=params)

    def close(self):
        try:
            self.ws.close()
        except Exception as e:
            print("WS error:", e)

class OgmiosQuery:
    def __init__(self, query, ogmios_connection=None):
        self.q = {"type": "jsonwsp/request","version": "1.0","servicename": "ogmios", "params": {}, "mirror": None, "jsonrpc": "2.0"}
        self.q.update(query)
        self.ogmios_connection = ogmios_connection

    def get(self, params={}):
        return json.dumps(dict(self.q, params=params))

    def run(self, params={}):
        req = self.get(params)
        m = self.ogmios_connection.send_recv(req)
        res = json.loads(m)
        return res
