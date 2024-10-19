# -*- coding: utf-8 -*-

import copy
import os
import sys
import math
import datetime

from pycardano.plutus import CBORTag
from pycardano.serialization import IndefiniteList
from pycardano.serialization import ByteString
import pycardano

class TunaState:
    def __init__(self, trie):
        self.state = {
                    'block': -1,
                    'hash': '',
                    'lz': 0,
                    'dn': 0,
                    'epoch': 0,
                    'posix_time': 0,
                    'merkle_root': None,
                    'trie': trie,
                    'miner': None,
                    'nonce': None,
                    'miner_cred_hash': None,
                    'miner_data': None,
                    }

    def get(self, k):
        return self.state.get(k)

    def update(self, tuna_tx):
        if (tuna_tx.out_block_number != self.state['block'] + 1) and self.state['block'] > 0:
            #print("wrong block number")
            return False

        new_trie = copy.deepcopy(self.state['trie'])
        new_trie.insert_digest(tuna_tx.out_current_hash)

        if new_trie.hash.hex() == tuna_tx.out_merkle_root:
            self.state.update({
                    'block': tuna_tx.out_block_number,
                    'hash': tuna_tx.out_current_hash,
                    'lz': tuna_tx.out_leading_zeros,
                    'dn': tuna_tx.out_target_number,
                    'epoch': tuna_tx.out_epoch_time,
                    'posix_time': tuna_tx.out_posix_time,
                    'merkle_root': tuna_tx.out_merkle_root,
                    'trie': new_trie,
                    'miner': tuna_tx.miner,
                    'nonce': tuna_tx.nonce,
                    'miner_cred_hash': tuna_tx.miner_cred_hash,
                    'miner_data': tuna_tx.miner_data,
                    })
            return True
        else:
            print("root does not match:", new_trie.hash.hex(), tuna_tx.out_merkle_root)
            return False

    def __repr__(self):
        fractional_difficulty = self.state['lz'] + (4-math.log(1+self.state['dn'])/math.log(16))
        info = [
            f"BLOCK:       {self.state['block']}",
            f"LZ/DN:       {self.state['lz']}/{self.state['dn']} => difficulty = {fractional_difficulty:.3f}",
            f"EPOCH TIME:  {self.state['epoch']} @ {datetime.datetime.fromtimestamp(self.state['posix_time']*0.001).isoformat()}",
            f"MERKLE ROOT: {self.state['merkle_root']}",
            ]
        return "\n".join(info)

