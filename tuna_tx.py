#!/usr/bin/env python3
import os

import pycardano
from hashlib import blake2b
from cardano_helpers import *


class TunaTxStatus:
    UNCONFIRMED = 0
    LATE = 1
    FAILED = 2
    SUCCESS = 3
    ERROR = 4
    ROLLEDBACK = 5

    _colorcodes = {
            0: '\x1b[100m',
            1: '\x1b[103m',
            2: '\x1b[41m',
            3: '\x1b[102m',
            4: '\x1b[101m',
            5: '\x1b[104m',
            }
    _names = {
            0: "unconfirmed",
            1: "late (UTxO spent)",
            2: "failed",
            3: "success",
            4: "error",
            5: "rolledback",
            }

    @staticmethod
    def codes():
        return list(TunaTxStatus._colorcodes.keys())

    @staticmethod
    def colorcode(status):
        return TunaTxStatus._colorcodes.get(status, '<x1b[106m')

    @staticmethod
    def name(status):
        return TunaTxStatus._names.get(status, f'unknown({status})')

class TunaTx:

    def __init__(self, tx, cbor=None, config={}):
        self.config = config

        self.TUNA_STATE_PREFIX_HEX   = self.config.get('TUNA_STATE_PREFIX', 'TUNA').encode('utf-8').hex()
        self.TUNA_COUNTER_PREFIX_HEX = self.config.get('TUNA_COUNTER_PREFIX','COUNTER').encode('utf-8').hex()

        if tx is None:
            self.tx = pycardano.Transaction.from_cbor(cbor)
        else:
            self.tx = tx
        self.cbor = cbor

        witness_set = self.tx.transaction_witness_set
        self.redeemers   = witness_set.redeemer

        # check validity
        if not self.tx.valid:
            raise Exception("failed tx")

        reference_scripts = self.tx.transaction_body.reference_inputs

        self.mint_script_idx = None
        self.spend_script_idx = None

        for idx,reference_input in enumerate(reference_scripts):
            if reference_input.transaction_id.to_primitive().hex() == self.config['MINT_SCRIPT_TX'] and reference_input.index == self.config['MINT_SCRIPT_IX']:
                self.mint_script_idx = idx
            if reference_input.transaction_id.to_primitive().hex() == self.config['SPEND_SCRIPT_TX'] and reference_input.index == self.config['SPEND_SCRIPT_IX']:
                self.spend_script_idx = idx
        if self.mint_script_idx is None or self.spend_script_idx is None:
            #raise Exception("reference scripts not found")
            pass

        for i,r in enumerate(self.redeemers):
            if r.tag == pycardano.plutus.RedeemerTag.MINT:
                self.mint_script_idx = i
            if r.tag == pycardano.plutus.RedeemerTag.SPEND:
                self.spend_script_idx = i

        try:
            self.mint_redeemer  = self.redeemers[self.mint_script_idx]
            self.spend_redeemer = self.redeemers[self.spend_script_idx]
        except:
            self.mint_redeemer  = None
            self.spend_redeemer = None
            self.miner_cred = None

        try:
            mint_redeemer_dict = self.mint_redeemer.data.to_dict()
            self.current_block = mint_redeemer_dict['fields'][1]['int']
        except:
            self.current_block = -1

        try:
            spend_redeemer_dict = self.spend_redeemer.data.to_dict()
            self.nonce = spend_redeemer_dict['fields'][0]['bytes']
            self.miner = spend_redeemer_dict['fields'][1]['fields'][0]['bytes']
            self.miner_cred = pycardano.plutus.RawPlutusData.from_dict(spend_redeemer_dict['fields'][1]).to_cbor()
            self.miner_cred_hash = blake2b(self.miner_cred, digest_size=32).digest().hex()
        except Exception as e:
            print("Error during tuna tx decoding:", e)
            self.nonce = None
            self.miner = None
            self.proof = None
            self.miner_cred = None
            self.miner_cred_hash = None

        self.tuna_out_datum = None
        tx_matches = False
        for tx_output in self.tx.transaction_body.outputs:

            assets_with_tuna_policy = get_assets_with_policy(tx_output, self.config['POLICY'])

            has_tuna_state   = any([x.startswith(self.TUNA_STATE_PREFIX_HEX) for x in assets_with_tuna_policy.keys()])
            has_tuna_counter = any([x.startswith(self.TUNA_COUNTER_PREFIX_HEX) for x in assets_with_tuna_policy.keys()])

            if has_tuna_state and has_tuna_counter:
                tx_matches = True
                self.tuna_out_datum = tx_output.datum
        if not tx_matches:
            raise Exception("state or counter not found.")

        out_datum_dict = self.tuna_out_datum.to_dict()

        self.out_block_number = out_datum_dict['fields'][0]['int']
        self.out_current_hash = out_datum_dict['fields'][1]['bytes']
        self.out_leading_zeros = out_datum_dict['fields'][2]['int']
        self.out_target_number = out_datum_dict['fields'][3]['int']
        self.out_epoch_time = out_datum_dict['fields'][4]['int']
        self.out_posix_time = out_datum_dict['fields'][5]['int']
        self.out_merkle_root = out_datum_dict['fields'][6]['bytes']


    def save(self, p):
        try:
            os.makedirs(p)
        except:
            pass

        with open(p + f'/{self.current_block}.tx', 'w') as f:
            f.write(self.cbor)
        print("saved:", p + f'/{self.current_block}.tx')

    def inspect(self):

        print("HEIGHT=", self.current_block)
        print("NONCE=", self.nonce)
        print("MINER=", self.miner)
        print("OUT_DATUM=", self.tuna_out_datum.to_dict())
        print("OUT_BLOCK_NUMBER=", self.out_block_number)
        print("OUT_HASH=", self.out_current_hash)
        print("OUT_LZ=", self.out_leading_zeros)
        print("OUT_TN=", self.out_target_number)
        print("OUT_EPOCHTIME=", self.out_epoch_time)
        print("OUT_POSIXTIME=", self.out_posix_time)
        print("OUT_MERKLE=", self.out_merkle_root)

