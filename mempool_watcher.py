import json
import cbor2
import time
import copy
import sys, traceback
from ogmios import Ogmios
from websockets.sync.client import connect
from threading import Thread, Event
from time import sleep
from pycardano import Transaction

from helpers import *
from cardano_helpers import *

from tuna_state import TunaState
from tuna_tx import *

class MempoolWatcher:
    def __init__(self, profile):
        self.profile = profile
        self.config = self.profile.config
        self.wallet = self.profile.wallet
        self.log = self.profile.log 
        self.ogmios = Ogmios(self.config.get('OGMIOS', 'ws://0.0.0.0:1337'), config=self.config) 
        self.running = False
        self.TUNA_STATE_PREFIX_HEX   = self.config.get('TUNA_STATE_PREFIX', 'TUNA').encode('utf-8').hex()
        self.TUNA_COUNTER_PREFIX_HEX = self.config.get('TUNA_COUNTER_PREFIX','COUNTER').encode('utf-8').hex()

        self.state = {
                'slot': 0,
                'wallet_utxos': [],
                'contract_utxos': [],
                'tuna': None,
                'tx': None,
                }

        self.has_state_update = Event()
        self.start()

    def start(self):
        if not self.running and self.config.get('USE_MEMPOOL', False):
            self.running = True
            self.thread = Thread(target=self.loop, args=())
            self.thread.start()

    def get_state(self):
        return self.state

    def is_possible_tuna_tx(self, cbor_hex):
        try:
            return self.config.get('POLICY') in cbor_hex and self.config.get('MINT_SCRIPT') in cbor_hex and self.config.get('SPEND_SCRIPT') in cbor_hex
        except:
            return False

    def deserialize_tuna_tx(self, cbor_hex):
        try:
            pyctx = pycardano.Transaction.from_cbor(cbor_hex)
            tx_spends_tuna_counter = False
            for tx_output in pyctx.transaction_body.outputs:
                assets_with_tuna_policy = get_assets_with_policy(tx_output, self.config['POLICY'])
                has_tuna_state   = any([x.startswith(self.TUNA_STATE_PREFIX_HEX) for x in assets_with_tuna_policy.keys()])
                has_tuna_counter = any([x.startswith(self.TUNA_COUNTER_PREFIX_HEX) for x in assets_with_tuna_policy.keys()])
                if has_tuna_state and has_tuna_counter:
                    tx_spends_tuna_counter = True
                    break
            if not tx_spends_tuna_counter:
                return None
            tuna_tx = TunaTx(pyctx, cbor=cbor_hex, config=self.config)
            return tuna_tx
        except Exception as e:
            self.log(f"error deserializing tuna tx: {e}")
        return None

    def try_state_update(self, tuna_tx):
        if self.state['tuna'] is None:
            return False

        try:
            success = self.state['tuna'].update(tuna_tx)
        except Exception as e:
            print(e)
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
        return success

    def loop(self):
        while True:
            self.tx_queue = []
            t0 = time.time()
            self.ogmios.query('AcquireMempool')
            new_valid_tuna_state_found = False
            while True:
                t0 = time.time()
                res = self.ogmios.query('NextTransaction', {'fields': 'all'})
                if res['result']['transaction'] is None:
                    break
                try:
                    tx = res['result']['transaction']

                    # check updated to utxos
                    if tx['spends'] == 'inputs':
                        new_wallet_utxos = []
                        for own_utxo in self.state['wallet_utxos']:
                            spent = False
                            if 'inputs' in tx:
                                for txi in tx['inputs']:
                                    if own_utxo['transaction']['id'] == txi['transaction']['id'] and own_utxo['index'] == txi['index']:
                                        spent = True
                                        break
                            if not spent:
                                new_wallet_utxos.append(own_utxo)
                        self.state['wallet_utxos'] = new_wallet_utxos
                        for i,txo in enumerate(tx['outputs']):
                            if txo['address'] == self.wallet:
                                self.state['wallet_utxos'].append({'transaction': {'id': tx['id']}, 'index': i, 'address': self.wallet, 'value': txo['value']})

                    # check tuna transactions
                    if 'cbor' not in tx:
                        self.log(f"<x1b[91m----------------------------------------------------------------------------------------------------<x1b[0m")
                        self.log(f"<x1b[91m----------------------------------------------------------------------------------------------------<x1b[0m")
                        self.log(f"<x1b[91merror: tx-object contains no CBOR. Ogmios from v6 on doesn't anymore include CBOR by default, enable it with --include-cbor<x1b[0m")
                        self.log(f"<x1b[91m----------------------------------------------------------------------------------------------------<x1b[0m")
                        self.log(f"<x1b[91m----------------------------------------------------------------------------------------------------<x1b[0m")
                    cbor_hex = tx['cbor']
                    if self.is_possible_tuna_tx(cbor_hex):
                        tuna_tx = self.deserialize_tuna_tx(cbor_hex)
                        if tuna_tx:
                            try:
                                success = self.try_state_update(tuna_tx)
                                if success:
                                    self.state['tx'] = tx['id']
                                new_valid_tuna_state_found = new_valid_tuna_state_found or success
                            except Exception as e:
                                self.log(f"state update failed: {e}")
                            
                            for i,txo in enumerate(tx['outputs']):
                                if txo['address'] == self.config['CONTRACT_ADDRESS']:
                                    self.state['contract_utxos'] = [{'transaction': {'id': tx['id']}, 'index': i, 'address': self.config['CONTRACT_ADDRESS'], 'value': txo['value']}]
                except Exception as e:
                    raise
            if new_valid_tuna_state_found:
                self.log("tuna state update in mempool")
                self.has_state_update.set()
