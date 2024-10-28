# -*- coding: utf-8 -*-
import json
import os
import pycardano
import datetime
import time
import copy
import sys, traceback

from ogmios_connection import OgmiosConnection
from threading import Thread, Event
from time import sleep
import copy

from pmtrie import *
from helpers import *
from cardano_helpers import *
from chain_index import ChainIndex

from tuna_state import TunaState
from tuna_tx import *

def posix_to_slots(pt, shelley_offset=1666656000):
    return (pt//1000) - shelley_offset

def slots_to_posix(slots, shelley_offset=1666656000):
    return (slots + shelley_offset)*1000

class ChainWatcher:
    def __init__(self, profile, ogmios_connection=None):
        self.profile = profile
        self.config  = self.profile.config
        self.log     = self.profile.log
        self.wallet  = self.profile.wallet
        if self.config.get('OGMIOS_SHARED_CONNECTION') and ogmios_connection is not None:
            self.ogmios = ogmios_connection
        else:
            self.ogmios = OgmiosConnection(self.config.get('OGMIOS', 'ws://0.0.0.0:1337'), config=self.config) 
        self.synced_with_time = False
        self.tx_debug = False
        self.submitted_transactions = []

        self.events = {
                'dirty': Event(),
                'synced': Event(),
                'has_tx': Event(),
                'rollback': Event(),
                'block': Event(),
                }

        self.state = {
                'contract_utxos': [],
                'wallet_utxos': [],
                'tuna': TunaState(self.trie_from_genesis()),
                'tx': None,
                }

        self.TUNA_STATE_PREFIX_HEX    = self.config.get('TUNA_STATE_PREFIX', 'TUNA').encode('utf-8').hex()
        self.TUNA_COUNTER_PREFIX_HEX  = self.config.get('TUNA_COUNTER_PREFIX','COUNTER').encode('utf-8').hex()
        self.TUNA_POLICY_HEX          = self.config['POLICY']
        self.TUNA_STATE_ASSETNAME_HEX = self.TUNA_STATE_PREFIX_HEX + self.config.get('SPEND_SCRIPT')
        self.TUNA_ASSETNAME_HEX       = self.config.get('TUNA_ASSETNAME').encode('utf-8').hex() 

        self.no_cbor_warning_shown = False

        self.restart()

    def is_alive(self):
        self.thread.join(timeout=0.0)
        return self.thread.is_alive()

    def restart(self):
        self.thread = Thread(target=self.loop, args=())
        self.thread.start()

    def trie_from_genesis(self):
        genesis_filename = f'./config/{self.profile.name}/genesis.json'
        if not os.path.isfile(genesis_filename):
            raise Exception(f"FileNotFound: {genesis_filename}")
        with open(genesis_filename, 'r') as gf:
            genesis = json.load(gf)

        t = PMtrie()
        if type(genesis) is dict:
            for k,v in genesis.items():
                t.insert(bytes.fromhex(k), bytes.fromhex(v))
        elif type(genesis) is list:
            for v in genesis:
                if type(v) is dict:
                    t.insert_digest(bytes.fromhex(v['current_hash']))
                else:
                    t.insert_digest(bytes.fromhex(v))
        else:
            self.log("<x1b[31merror: unknown format for merkle trie genesis<x1b[0m")
            os._exit(5)

        self.log(f"merkle root: {t.hash.hex()}")
        return t

    def add_submitted_tx(self, tx):
        self.submitted_transactions.append(tx)

    def check_submitted_tx_status(self):
        if len(self.submitted_transactions) < 1:
            return '?'

        for status_code in TunaTxStatus.codes():
            n_tx = len(list(filter(lambda x: x[2] == status_code, self.submitted_transactions[-50:])))
            self.log(f"{TunaTxStatus.name(status_code)} ({n_tx}) ".rjust(24) + f": {TunaTxStatus.colorcode(status_code)}{' '*n_tx}<x1b[0m")

        status_string = ''.join(TunaTxStatus.colorcode(h_tx[2]) + ' \x1b[0m' for h_tx in self.submitted_transactions[-50:])
        self.log(f"history: [{status_string}]")

    def query(self, query_name, params={}):
        return self.ogmios.query(query_name, params=params)

    def query_utxos(self):
        success = False
        try:
            self._query_utxos()
            success = True
        except Exception as e:
            self.log(f"<x1b[31merror: could not query utxos: {e}<x1b[0m")
        if success:
            return

        try:
            self.log(f"try reconnecting to Ogmios...")
            time.sleep(5)
            self.ogmios  = Ogmios(self.config.get('OGMIOS', 'ws://0.0.0.0:1337'), config=self.config)
            self._query_utxos()
            success = True
        except Exception as e:
            self.log(f"<x1b[31merror: could not query utxos (failed again): {e}<x1b[0m")


    def _query_utxos(self):
        t0 = time.time()
        utxos = self.query('LocalStateQuery', {'addresses': [self.wallet, self.config['CONTRACT_ADDRESS']]})['result']
        if time.time()-t0 > 4:
            self.log(f"performance warning: query wallet and contract utxos took too long: {(time.time()-t0):.1f}s")

        self.state['wallet_utxos']   = list(filter(lambda x: x['address'] == self.wallet, utxos))
        self.state['contract_utxos'] = list(filter(lambda x: x['address'] == self.config['CONTRACT_ADDRESS'], utxos))

    def get_state(self):
        return self.state

    # loose pre-filter to reject most tx without more expensive tx deserialization
    def is_possible_tuna_tx(self, tx):
        if not self.no_cbor_warning_shown and tx is not None and 'cbor' not in tx:
            self.no_cbor_warning_shown = True
            self.log(f"<x1b[91m----------------------------------------------------------------------------------------------------<x1b[0m")
            self.log(f"<x1b[91m----------------------------------------------------------------------------------------------------<x1b[0m")
            self.log(f"<x1b[91merror: tx-object contains no CBOR. Ogmios from v6 on doesn't anymore include CBOR by default, enable it with --include-cbor<x1b[0m")
            self.log(f"<x1b[91m----------------------------------------------------------------------------------------------------<x1b[0m")
            self.log(f"<x1b[91m----------------------------------------------------------------------------------------------------<x1b[0m")
        try:
            cbor_hex = tx['cbor']
            return self.config.get('POLICY') in cbor_hex and self.config.get('MINT_SCRIPT') in cbor_hex and self.config.get('SPEND_SCRIPT') in cbor_hex
        except Exception as e:
            print(e)
            return False

    def try_state_update(self, tuna_tx, tx=None):
        success = False
        try:
            success = self.state['tuna'].update(tuna_tx)
        except Exception as e:
            print(e)
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)

        if success:
            time_behind = format_timedelta_since_posix_time(tuna_tx.out_posix_time)
            if len(time_behind) < 1:
                self.log(f"<x1b[96m\U0001F41F {tuna_tx.out_block_number} {datetime.datetime.fromtimestamp(tuna_tx.out_posix_time * 0.001).isoformat()} (synced) <x1b[0m")
            else:
                self.log(f"<x1b[96m\U0001F41F {tuna_tx.out_block_number} {datetime.datetime.fromtimestamp(tuna_tx.out_posix_time * 0.001).isoformat()} ({time_behind} behind) <x1b[0m")

            try:
                self.state['tuna_tx_cbor'] = tuna_tx.cbor
            except:
                self.state['tuna_tx_cbor'] = None

            try:
                self.state['tx'] = tx['id']
            except:
                pass

            try:
                self.index_state() 
            except Exception as e:
                print("error during store to db:", e)
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
        else:
            if tuna_tx.out_block_number != self.state['tuna'].state['block']:
                self.log(f"<x1b[91minvalid state transition to block height {tuna_tx.out_block_number}<x1b[0m")
        return success

    def index_state(self):
        # Cardano state
        record = {
                'block': self.synced_tip['height'],
                'slot': self.synced_tip['slot'],
                'id': self.synced_tip['id'],
                'tx': self.state['tx'],
                }
        # TUNA state with prefix 'tuna_'
        tuna_state_columns = ['block', 'hash', 'lz', 'dn', 'epoch', 'posix_time', 'merkle_root', 'miner', 'nonce', 'miner_cred_hash', 'miner_data']
        for c in tuna_state_columns:
            record['tuna_' + c] = self.state['tuna'].state.get(c)

        record['cbor'] = bytes([])
        try:
            if self.state['tuna_tx_cbor'] is not None:
                record['cbor'] = bytes.fromhex(self.state['tuna_tx_cbor'])
        except Exception as e:
            print(e)

        self.db.insert(record)

    def deserialize_tuna_tx(self, tx, debug=False):
        try:
            cbor_hex = tx['cbor']
            pyctx = pycardano.Transaction.from_cbor(cbor_hex)
            if self.tx_debug or debug:
                print(pyctx.transaction_body)

            tx_spends_tuna_counter = False
            for tx_output in pyctx.transaction_body.outputs:
                assets_with_tuna_policy = get_assets_with_policy(tx_output, self.TUNA_POLICY_HEX)
                has_tuna_state   = any([x.startswith(self.TUNA_STATE_PREFIX_HEX) for x in assets_with_tuna_policy.keys()])
                has_tuna_counter = any([x.startswith(self.TUNA_COUNTER_PREFIX_HEX) for x in assets_with_tuna_policy.keys()])
                if has_tuna_state and has_tuna_counter:
                    tx_spends_tuna_counter = True
                    break

            if not tx_spends_tuna_counter:
                return None

            tuna_tx = TunaTx(pyctx, cbor=cbor_hex, config=self.config)
            if debug:
                tuna_tx.inspect()
            return tuna_tx
        except Exception as e:
            self.log(f"error deserializing tuna tx: {e}")
        return None

    def update_sync_status(self, block):

        self.slot       = block['slot']
        self.state['slot'] = self.slot
        self.synced_tip = {'slot': self.slot, 'height': block['height'], 'id': block['id']}

        if self.synced_tip['slot'] >= self.network_tip['slot']:
            self.network_tip = self.query('QueryNetworkTip')['result']

            if self.synced_tip['slot'] >= self.network_tip['slot'] and self.synced_with_time:
                if not self.synced:
                    self.synced = True
                    self.query_utxos()
                    self.events['block'].set()
                    self.events['synced'].set()
                    self.log("-"*80)
                    self.log("reached the tip.")
                    self.log(f"slot:   {self.tip['slot']}")
                    self.log(f"height: {self.tip['height']}")
                    self.log(f"id:     {self.tip['id']}")
                    self.log("-"*80)
            self.tip = self.synced_tip 

    def loop(self):

        self.db = ChainIndex(self.profile, self)
        self.log(f"chain indexer initialized: {self.db}")

        self.network_tip = self.query('QueryNetworkTip')['result']

        success = False
        while not success:
            try:
                block1 = self.query('NextBlock')
                success = True
            except Exception as e:
                print(e)
                print("error connecting to node, retrying...")
                sleep(5)

        tip = block1["result"]["tip"]
        self.log(f"connected to the chain, tip is: {tip}")
        self.tip = copy.deepcopy(tip)

        state = self.db.get_state()
        if state is not None:
            self.log(f"found a snapshot, syncing from snapshot")

            # replay chain from DB to fill trie
            chain = self.db.get_chain()
            for row in chain:
                self.state['tuna'].state['trie'].insert_digest(row['tuna_hash'])
                if self.state['tuna'].state['trie'].hash.hex() != row['tuna_merkle_root']:
                    self.log(f"<x1b[31mdatabase errror: merkle root validation failed at block {row['tuna_block']}<x1b[0m")
                    os._exit(1)

            # set state from DB
            self.state['tuna'].state.update({
                'block': state['tuna_block'],
                'hash': state['tuna_hash'],
                'lz': state['tuna_lz'],
                'dn': state['tuna_dn'],
                'epoch': state['tuna_epoch'],
                'posix_time': state['tuna_posix_time'],
                'merkle_root': state['tuna_merkle_root'],
                })

            self.sync_from = {
                    'slot': state['slot'],
                    'id': state['id'],
                    'height': state['block'],
                    }

            self.log(f"starting from block {self.state['tuna'].state['block']}")
            
        else:
            self.log(f"syncing from genesis.")
            self.sync_from = {
                    'slot': self.config.get("SYNC_SLOT"), 
                    'id': self.config.get("SYNC_HASH"),
                    'height': self.config.get("SYNC_BLOCKNO"),
                    }
            self.log(f"starting from genesis")

        if self.sync_from:
            self.log(f"sync from: {self.sync_from}")
            m = self.query('FindIntersect', {'points': [self.sync_from]})
            self.synced_tip = m['result']['intersection']
            sleep(1)
            self.synced = False
        else:
            self.query('FindIntersect', {'points': [self.tip]})

        last_sync_update = time.monotonic()-1
        last_sync_slot = -1
        first = True
        while True:
            block_update = self.query('NextBlock')['result']

            direction = block_update.get('direction')
            if direction == 'forward':
                block = block_update['block'] 

                if not self.synced:
                    if time.monotonic() > last_sync_update + 5:
                        speed = f"({(block['slot']-last_sync_slot)/(time.monotonic()-last_sync_update):.0f} x)" if last_sync_slot > 0 else ''
                        self.log(f"syncing, block height {block['height']} {speed}")
                        last_sync_update = time.monotonic()
                        last_sync_slot = block['slot']

                time_now = int(datetime.datetime.now().timestamp())*1000
                time_now_in_slots = posix_to_slots(time_now, self.config.get('SHELLEY_OFFSET'))
                self.synced_with_time = block['slot'] > time_now_in_slots - 91

                new_valid_tuna_state_found = False
                for tx in block.get('transactions', []):
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

                    if self.is_possible_tuna_tx(tx):
                        tuna_tx = self.deserialize_tuna_tx(tx)
                        if tuna_tx:
                            try:
                                success = self.try_state_update(tuna_tx, tx=tx)
                                new_valid_tuna_state_found = new_valid_tuna_state_found or success
                            except Exception as e:
                                self.log(f"state update failed: {e}")
                            
                            for i,txo in enumerate(tx['outputs']):
                                if txo['address'] == self.config['CONTRACT_ADDRESS']:
                                    self.state['contract_utxos'] = [{'transaction': {'id': tx['id']}, 'index': i, 'address': self.config['CONTRACT_ADDRESS'], 'value': txo['value']}]

                            for s_tx in self.submitted_transactions:
                                if s_tx[0] == tuna_tx.out_block_number:
                                    if s_tx[1] == tx['id']:
                                        s_tx[2] = 3
                                    elif s_tx[2] != 2:
                                        s_tx[2] = 1

                if new_valid_tuna_state_found:
                    if self.synced:
                        self.log("mining...")
                        self.events['block'].set()

                self.update_sync_status(block)
                

            elif direction == 'backward':
                self.log("<x1b[30m<x1b[43m@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ROLLBACK @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@<x1b[0m")
                self.log(f"--> {block_update['tip']}")
                rows_deleted = self.db.rollback(block_update['tip']['height'])
                if rows_deleted > 0:
                    self.log(f"deleted {rows_deleted} rows")

                t0 = time.monotonic()
                # re-init 
                self.state['tuna'] = TunaState(self.trie_from_genesis())

                # replay chain from DB to fill trie
                chain = self.db.get_chain()
                for row in chain:
                    self.state['tuna'].state['trie'].insert_digest(row['tuna_hash'])
                    if self.state['tuna'].state['trie'].hash.hex() != row['tuna_merkle_root']:
                        self.log(f"<x1b[31mdatabase error: merkle root validation failed at block {row['tuna_block']}<x1b[0m")
                        os._exit(1)

                # set state from DB
                state = self.db.get_state()
                if state is None or 'tuna_block' not in state:
                    self.log(f"<x1b[31mdatabase error: invalid tuna state, if this error persists, check genesis file and network. re-syncing...<x1b[0m") 
                    continue

                self.state['tuna'].state.update({
                    'block': state['tuna_block'],
                    'hash': state['tuna_hash'],
                    'lz': state['tuna_lz'],
                    'dn': state['tuna_dn'],
                    'epoch': state['tuna_epoch'],
                    'posix_time': state['tuna_posix_time'],
                    'merkle_root': state['tuna_merkle_root'],
                    })
                t1 = time.monotonic()
                self.log(f"re-initialized Patricia Merkle Trie, took {(t1-t0):.2f} seconds, at: {self.state['tuna'].state['block']}")
                self.log('-'*80)
                for x in repr(self.state['tuna']).split('\n'):
                    self.log(x)
                self.log('-'*80)

                self.query_utxos()
                self.update_sync_status(block_update['tip'])
                self.log(f"<continue>")

            if self.events['dirty'].is_set():
                self.events['dirty'].clear()
                self.query_utxos()
                self.log(f"<updated>")

