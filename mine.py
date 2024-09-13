#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import copy
import datetime
import time
import random
import hashlib
import subprocess
import pycardano
import socket
import math
import shutil

from pycardano.plutus import CBORTag
from pycardano.plutus import RawPlutusData

from ogmios import Ogmios
from chain_watcher import ChainWatcher
from mempool_watcher import MempoolWatcher
from config_profile import Profile
from cardano_helpers import *
from target_state import TargetState
from miners import MinerManager, format_hashrate
from tuna_tx import *

PROGRAM_NAME         = "bigeye"
VERSION              = "v0.2.4"

parser = argparse.ArgumentParser(
                    prog=f'{PROGRAM_NAME} {VERSION}',
                    description='Cardano $TUNA miner',
                    epilog='')
parser.add_argument('profile', default='preview')
args = parser.parse_args()

# profile and config
try:
    profile = Profile(args.profile)
    config = profile.config 
    logger = profile.log
except Exception as e:
    print(f"failed loading profile: {args.profile}, error: {e}")
    exit(2)


logger('-'*80)
logger(f"initialized {PROGRAM_NAME} {VERSION}")
logger('-'*80)
logger(f'NETWORK: {config.get("NETWORK")}')
logger(f'WALLET:  {profile.wallet}')
logger(f'POLICY:  {profile.TUNA_POLICY_HEX}')
logger('-'*80)

# ogmios for tx submission

OGMIOS_URL = config.get('OGMIOS', 'ws://0.0.0.0:1337')
ogmios = None
while ogmios is None:
    try:
        ogmios = Ogmios(OGMIOS_URL, config=config)
    except Exception as e:
        logger(f"<x1b[91merror: failed to connect to ogmios @{OGMIOS_URL}: {e}")
        time.sleep(5)

logger(f"connected to ogmios@{OGMIOS_URL}")

# protocol parameters
try:
    protocol_parameters = ogmios.query('ProtocolParameters')
    PLUTUS_V2_COST_MODEL = {i: v for i,v in enumerate(protocol_parameters['result']['plutusCostModels']['plutus:v2'])}
except Exception as e:
    logger(f"<x1b[41merror: could not obtain protocol parameters, is cardano-node running? Error from ogmios: {e}<x1b[0m")
    exit()


# watch chain and mempool
chain   = ChainWatcher(profile=profile)
mempool = MempoolWatcher(profile=profile)


MINERS = profile.config.get('MINERS', '127.0.0.1:2023').split(',')

# AUTO_SPAWN_MINERS
AUTO_SPAWN_MINERS = config.get('AUTO_SPAWN_MINERS', False)
miner_processes = []
if AUTO_SPAWN_MINERS:
    MINER_EXECUTABLE = config.get('MINER_EXECUTABLE', "miners/simple/miner_core.py")
    if shutil.which(MINER_EXECUTABLE) is None:
        logger(f"<x1b[41merror: miner executable {MINER_EXECUTABLE} not found<x1b[0m")
        if MINER_EXECUTABLE in ["miners/cpu/cpu-sha256", "./miners/cpu/cpu-sha256"]:
            logger(f"hint: to build the CPU miner, go to the ./miners/cpu directory and run `make`")
        os._exit(3)

    for m in MINERS:
        HOST = m.split(':')[0]
        port_def = m.split(':')[1]
        if '-' in port_def:
            ports = list(range(int(port_def.split('-')[0]), int(port_def.split('-')[1])+1))
        else:
            ports = [int(port_def)]
        for PORT in ports:
            # only localhost allowed
            miner_processes.append(subprocess.Popen([MINER_EXECUTABLE, str(PORT)], stdout=subprocess.DEVNULL))
            logger(f"spawned a new miner process :{str(PORT)}, now running {len(miner_processes)}")


# example for multiple miners on consecutive ports: 127.0.0.1:2023-2038
miners = MinerManager(profile=profile)
for m in MINERS:
    if len(m.strip()) < 1:
        continue
    HOST = m.split(':')[0]
    try:
        port_def = m.split(':')[1]
    except:
        port_def = '2023'
    if '-' in port_def:
        ports = list(range(int(port_def.split('-')[0]), int(port_def.split('-')[1])+1))
    else:
        ports = [int(port_def)]
    for PORT in ports:
        miners.add(config={'HOST': HOST, 'PORT': PORT})
logger(f"initialized {len(miners)} miners.")

VERSION_INFO         = f"{PROGRAM_NAME} {VERSION}"
TUNA_POLICY          = profile.TUNA_POLICY_HEX
TUNA_STATE_ASSETNAME = profile.TUNA_STATE_ASSETNAME_HEX
TUNA_COUNTER_PREFIX  = profile.TUNA_COUNTER_PREFIX_HEX
TUNA_ASSETNAME       = profile.TUNA_ASSETNAME_HEX
CONTRACT_ADDRESS     = profile.CONTRACT_ADDRESS
OWN_ADDRESS          = profile.wallet
WALLET_SKEY          = profile.wallet_skey
WALLET_VKEY          = profile.wallet_vkey
TX_FEE               = config.get('TX_FEE', 685000)
EPOCH_NUMBER         = config.get('EPOCH_NUMBER', 2016)
EPOCH_TARGET         = config.get('MILLISECONDS_PER_EPOCH', 600000) * EPOCH_NUMBER
MINER_TARGET_STATE_REFRESH_INTERVAL = config.get('MINER_TARGET_STATE_REFRESH_INTERVAL', 14)
POSIX_TIME_DELTA     = config.get('POSIX_TIME_DELTA', 85000)
MAX_WAIT_UNTIL_VALID_SECONDS = config.get('MAX_WAIT_UNTIL_VALID_SECONDS', 90)
GLOBAL_TIMEOUT       = config.get('GLOBAL_TIMEOUT', 999*356*86400)

WAIT_FOR_BLOCKS_S  = 0.1
WAIT_FOR_MEMPOOL_S = 0.1

state = {
        'confirmed': None,
        'speculative': None,
        }

time_histogram_node = [0]*11
time_histogram_now  = [0]*11
additional_lovelaces_to_contract = 0
submitted_blocks = []
last_time_between_blocks_estimation_time = time.monotonic()
solution_submitted_timeout = -1
last_success_time = time.monotonic()

while True:

    # check for new blocks
    has_block = chain.events['block'].wait(timeout=WAIT_FOR_BLOCKS_S)
    if has_block:
        # tuna block!
        solution_submitted_timeout = -1
        chain.events['block'].clear()
        state['confirmed'] = chain.get_state()

        mempool.state = copy.deepcopy(chain.state)

    # check for new tx in mempool
    has_mempool_tx = mempool.has_state_update.wait(timeout=WAIT_FOR_MEMPOOL_S)
    if has_mempool_tx:
        # tuna block!
        solution_submitted_timeout = -1
        mempool.has_state_update.clear()
        state['speculative'] = mempool.get_state()

    # restart mining from confirmed state
    if solution_submitted_timeout > 0 and time.time() > solution_submitted_timeout:
        solution_submitted_timeout = -1
        chain.events['dirty'].set()
        state['confirmed'] = chain.get_state()
        logger(f"reset state to previous head.")


    # todo: improve logic to select what to mine
    use_state = 'speculative' if state['speculative'] is not None else 'confirmed' 
    if state[use_state]:

        slot = state[use_state]['slot']
        collateral_utxo = get_collateral_utxo(state[use_state]['wallet_utxos'])
        own_input_utxo = get_tx_input_utxo(state[use_state]['wallet_utxos'])
        contract_utxos = state[use_state]['contract_utxos']
        tuna_state = state[use_state]['tuna']

        in_block = tuna_state.get('block')
        in_hash = tuna_state.get('hash')
        in_epoch = tuna_state.get('epoch')
        in_posix_time = tuna_state.get('posix_time')
        in_lz = tuna_state.get('lz') 
        in_dn = tuna_state.get('dn')

        out_block = in_block + 1
        time_now = int(datetime.datetime.now().timestamp())*1000 - POSIX_TIME_DELTA
        out_epoch = in_epoch + 90000 + time_now - in_posix_time
        out_posix_time = 90000 + time_now

        ts = TargetState(WALLET_VKEY.hash(), VERSION_INFO)
        ts.epoch_time    = in_epoch 
        ts.block_number  = in_block
        ts.current_hash  = bytes.fromhex(in_hash)
        ts.leading_zeros = in_lz
        ts.target_number = in_dn

        if config.get('_USE_PREVIEW_V1_SERIALIZATION'):
            ts._use_preview_v1_serialization = True

        tsdata = ts.serialize()

        miners.set_difficulty(in_lz, in_dn)
        miners.set_target(tsdata.hex())

        # MINING
        t_start = time.monotonic()
        found_solution = False
        while time.monotonic() < t_start + MINER_TARGET_STATE_REFRESH_INTERVAL:

            result = miners.wait()
            if result:
                try:
                    nonce = bytes.fromhex(result)

                    tsdata = tsdata[:4] + nonce + tsdata[20:]
                    dh = hashlib.sha256(hashlib.sha256(tsdata).digest()).digest()
                    this_lz, this_dn = calc_diff(dh)
                    if this_lz > in_lz or ( this_lz == in_lz and this_dn < in_dn):
                        logger(f"<x1b[95mMINER: found a solution with {this_lz} {this_dn}, difficulty is {in_lz} {in_dn}<x1b[0m")
                        found_solution = True
                        break
                except Exception as e:
                    logger(f"<x1b[91merror: invalid response from miner: {result} => {e}")
            
            stats = miners.get_stats()

            n_miners       = len(stats.keys()) 
            active_miners  = sum(1 if s['rate'] > 0 else 0 for k,s in stats.items())
            total_hashrate = sum(s['rate'] for k,s in stats.items())

            if active_miners < n_miners:
                logger(f"\U0001F3A3 total fishing rate: {format_hashrate(total_hashrate)} <x1b[41mactive: {active_miners}/{n_miners}<x1b[0m")
            else:
                logger(f"\U0001F3A3 total fishing rate: {format_hashrate(total_hashrate)} active: {active_miners}/{n_miners}")

            # expected time to find solution estimate and difficulty change prediction
            if time.monotonic() > last_time_between_blocks_estimation_time + 120:
                last_time_between_blocks_estimation_time = time.monotonic()
                try:
                    n_hashes_per_solution = 16**in_lz * (65536/in_dn)
                    seconds_between_blocks = n_hashes_per_solution / total_hashrate
                    next_diff_change = EPOCH_NUMBER - (in_block % EPOCH_NUMBER)
                    expected_block_time = EPOCH_TARGET / EPOCH_NUMBER
                    direction = '??'
                    try:
                        if (in_block % EPOCH_NUMBER) > 5:
                            time_per_block = in_epoch / (in_block % EPOCH_NUMBER)
                            time_remaining = next_diff_change * (time_per_block/1000)
                            if time_per_block > expected_block_time:
                                factor = min(4, time_per_block/expected_block_time)
                                direction = f'down {factor:.1f}x in {format_seconds(time_remaining)}'
                            else:
                                factor = min(4, expected_block_time/time_per_block)
                                direction = f'up {factor:.1f}x in {format_seconds(time_remaining)}'
                    except:
                        pass
                    logger(f"\u23F1  estimated time to next solution: <x1b[92m{format_seconds(seconds_between_blocks)}<x1b[0m.")
                    logger(f"\U0001F41F at height {in_block}: LZ={in_lz} DN={in_dn}, next difficulty change in {next_diff_change} blocks, expected direction: {direction}")
                    logger(f"\u26D3  Cardano at slot: {slot}")

                    lovelaces_remaining = lovelace_value_from_utxos(state[use_state]['wallet_utxos'])
                    currency_symbol = "\u20B3" if config.get('NETWORK', 'MAINNET').upper() == 'MAINNET' else "t\u20B3"

                    tuna_value = token_value_from_utxos(state[use_state]['wallet_utxos'], TUNA_POLICY, TUNA_ASSETNAME)

                    if lovelaces_remaining < 25000000:
                        logger(f"\U0001F4B0 <x1b[93m{lovelaces_remaining/1000000:.2f} {currency_symbol} wallet balance low!<x1b[0m , \U0001F41F {tuna_value/1e8:.0f}")
                    else:
                        logger(f"\U0001F4B0 {lovelaces_remaining/1000000:.2f} {currency_symbol}, \U0001F41F {tuna_value/1e8:.0f}")

                except:
                    logger(f"could not estimate time to next solution: LZ={in_lz} DN={in_dn} hash_rate={total_hashrate} epochs={EPOCH_NUMBER}")

            if time.monotonic() > last_success_time + GLOBAL_TIMEOUT:
                logger(f"<x1b[93mno solution found for a long time, terminating<x1b[0m")
                os._exit(10)

        # allow update of chain state if no solution found so far
        if not found_solution:
            continue

        out_block_hash = dh.hex() 

        # difficulty adjustment
        if in_block % EPOCH_NUMBER == 0 and in_block > 0:
            total_epoch_time = in_epoch + 90000 + time_now - in_posix_time
            adjustment_numerator, adjustment_denominator = get_difficulty_adjustment(total_epoch_time, EPOCH_TARGET)
            out_dn, out_lz = get_new_difficulty(in_dn, in_lz, adjustment_numerator, adjustment_denominator)
            out_epoch = 0
            logger(f"<x1b[44mDIFFICULTY ADJUSTMENT --> LZ={out_lz} DN={out_dn}<x1b[0m")
        else:
            out_lz = in_lz
            out_dn = in_dn

        new_trie = copy.deepcopy(tuna_state.get('trie'))
        new_trie.insert_digest(out_block_hash)

        proof = new_trie.prove_digest(out_block_hash).toPycardanoCBOR()
        out_merkle_root = new_trie.hash.hex()

        def counter_assetname(block):
            hex_counter = hex(block)[2:]
            if len(hex_counter) % 2 != 0:
                hex_counter = '0' + hex_counter
            return TUNA_COUNTER_PREFIX + hex_counter

        counter_in_assetname = counter_assetname(in_block)
        counter_out_assetname = counter_assetname(out_block) 

        contract_in_utxos = list(filter(lambda x: x['value'].get(TUNA_POLICY, {}).get(counter_in_assetname, 0) > 0 and x['value'].get(TUNA_POLICY, {}).get(TUNA_STATE_ASSETNAME, 0) > 0, contract_utxos))
        if len(contract_in_utxos) > 1:
            print("contract input ambiguous")
            continue
        elif len(contract_in_utxos) < 1:
            print("contract input not found:", counter_in_assetname, " in ", contract_utxos)
            chain.events['dirty'].set()
            continue
        contract_in_utxo = contract_in_utxos[0]

        contract_in_coin = value_from_utxo(contract_in_utxo).coin
        contract_out_coin = contract_in_coin + additional_lovelaces_to_contract

        # INPUTS --------------------------------------------------------------------------------------------------------------
        tx_inputs = []
        tx_inputs.append(tx_input_from_utxo(contract_in_utxo))
        tx_inputs.append(tx_input_from_utxo(own_input_utxo))

        # redeemer index
        tx_input_list = [[str(x.transaction_id), x.index] for x in tx_inputs]
        tx_input_list.sort()
        contract_input_index = tx_input_list.index([str(contract_in_utxo['transaction']['id']), contract_in_utxo['index']])

        # COLLATERAL ----------------------------------------------------------------------------------------------------------
        total_collateral     = 5000000
        collateral_in        = tx_input_from_utxo(collateral_utxo)
        if collateral_in is None:
            logger(f"<x1b[37m<x1b[41mno collateral UTXo found<x1b[0m")
            continue
        collateral           = [collateral_in]
        collateral_in_value  = value_from_utxo(collateral_utxo)
        collateral_out_value = collateral_in_value - pycardano.transaction.Value(coin=total_collateral)
        collateral_return    = pycardano.transaction.TransactionOutput(pycardano.address.Address.from_primitive(OWN_ADDRESS), collateral_out_value)

        # REFERENCE INPUTS ----------------------------------------------------------------------------------------------------
        reference_inputs = [
                tx_input_from_utxo({'transaction': {'id': config.get('SPEND_SCRIPT_TX')}, 'index': config.get('SPEND_SCRIPT_IX')}),
                tx_input_from_utxo({'transaction': {'id': config.get('MINT_SCRIPT_TX')}, 'index': config.get('MINT_SCRIPT_IX')}),
                ]

        # FEE -----------------------------------------------------------------------------------------------------------------
        tx_fee = TX_FEE

        # MINT -------------------------------------------------------------------------------------------------------------
        block_reward = 5000000000 #TODO
        mint = pycardano.transaction.MultiAsset({
            pycardano.ScriptHash(bytes.fromhex(TUNA_POLICY)): pycardano.transaction.Asset({
                    pycardano.transaction.AssetName(bytes.fromhex(TUNA_ASSETNAME)): block_reward,
                    pycardano.transaction.AssetName(bytes.fromhex(counter_out_assetname)): 1,
                    pycardano.transaction.AssetName(bytes.fromhex(counter_in_assetname)): -1,
                })
            })

        # OUTPUTS -------------------------------------------------------------------------------------------------------------
        tx_outputs = []
        contract_out_amount = pycardano.transaction.Value(coin=contract_out_coin, multi_asset=pycardano.transaction.MultiAsset({
                    pycardano.ScriptHash(bytes.fromhex(TUNA_POLICY)): pycardano.transaction.Asset({
                        pycardano.transaction.AssetName(bytes.fromhex(TUNA_STATE_ASSETNAME)): 1,
                        pycardano.transaction.AssetName(bytes.fromhex(counter_out_assetname)): 1
                        }),
                    }))
        contract_out_datum = RawPlutusData.from_primitive(
                CBORTag(121, [
                    out_block,
                    bytes.fromhex(out_block_hash),
                    out_lz,
                    out_dn,
                    out_epoch,
                    out_posix_time,
                    bytes.fromhex(out_merkle_root)
                ] ))
        contract_output = pycardano.transaction.TransactionOutput(pycardano.address.Address.from_primitive(CONTRACT_ADDRESS), contract_out_amount, datum=contract_out_datum)

        own_input_value   = value_from_utxo(own_input_utxo)
        fee_value         = pycardano.transaction.Value(coin=tx_fee)
        minted_tuna_value = pycardano.transaction.Value.from_primitive([0, {bytes.fromhex(TUNA_POLICY): {bytes.fromhex(TUNA_ASSETNAME): block_reward}}])

        own_output_value = own_input_value - fee_value + minted_tuna_value - additional_lovelaces_to_contract
        own_output       = pycardano.transaction.TransactionOutput(pycardano.address.Address.from_primitive(OWN_ADDRESS), own_output_value)

        if additional_lovelaces_to_contract > 0:
            additional_lovelaces_to_contract = 0

        tx_outputs.append(own_output)
        tx_outputs.append(contract_output)

        # REDEEMERS -------------------------------------------------------------------------------------------------------------
        redeemer_datum_spend = RawPlutusData(CBORTag(121, [nonce, ts.miner_credential, proof]))
        redeemer_datum_mint = RawPlutusData(CBORTag(122, [CBORTag(121, [CBORTag(121, [bytes.fromhex(contract_in_utxo['transaction']['id'])]), contract_in_utxo['index']]), in_block]))

        redeemers = [
                pycardano.Redeemer(redeemer_datum_spend, ex_units=pycardano.ExecutionUnits(900000, 400000000)),
                pycardano.Redeemer(redeemer_datum_mint, ex_units=pycardano.ExecutionUnits(180000, 90000000)),
                ]
        redeemers[0].tag = pycardano.plutus.RedeemerTag(0) #SPEND
        redeemers[1].tag = pycardano.plutus.RedeemerTag(1) #MINT
        redeemers[0].index = contract_input_index 
        redeemers[1].index = 0

        # SCRIPT DATA HASH ------------------------------------------------------------------------------------------------------
        cost_models      = pycardano.plutus.CostModels({1: PLUTUS_V2_COST_MODEL})
        script_data_hash = pycardano.utils.script_data_hash(redeemers, None, cost_models)

        # TIME VALIDITY ---------------------------------------------------------------------------------------------------------
        validity_start = posix_to_slots(out_posix_time, config.get('SHELLEY_OFFSET')) - 90
        ttl            = posix_to_slots(out_posix_time, config.get('SHELLEY_OFFSET')) + 90


        # BODY
        tx_body = pycardano.TransactionBody(
                inputs=tx_inputs,
                outputs=tx_outputs,
                mint=mint,
                fee=tx_fee,
                reference_inputs=reference_inputs,
                collateral=collateral,
                validity_start=validity_start,
                ttl=ttl,
                script_data_hash=script_data_hash,
                total_collateral=total_collateral,
                collateral_return=collateral_return,
                required_signers=[WALLET_VKEY.hash()],
                )

        # WITNESS
        tx_hash = tx_body.hash()
        signature = WALLET_SKEY.sign(tx_hash)
        vkey_witness = pycardano.witness.VerificationKeyWitness(WALLET_VKEY, signature)
        transaction_witness_set = pycardano.witness.TransactionWitnessSet(
                vkey_witnesses=[vkey_witness],
                redeemer=redeemers,
                )

        # TRANSACTION
        tx = pycardano.transaction.Transaction(
                transaction_body=tx_body,
                transaction_witness_set=transaction_witness_set
                )

        # WAIT?
        wait_until_valid = validity_start - slot
        if wait_until_valid > 0:
            logger(f"<x1b[93mwarning: build a tx, but node is lagging back by {wait_until_valid:.1f}s making the tx invalid<x1b[0m")
            logger(f"waiting up to {MAX_WAIT_UNTIL_VALID_SECONDS} s for cardano-node to catch up...")
            t_start = time.monotonic()
            while time.monotonic() < t_start + MAX_WAIT_UNTIL_VALID_SECONDS: 
                time.sleep(1)
                slot = chain.get_state()['slot']
                if slot >= validity_start:
                    break

            if slot >= validity_start:
                logger(f"<x1b[96mrecovered: tx is valid now, submitting...<x1b[0m")
            else:
                logger(f"<x1b[43mwarning: tx probably still not valid, but MAX_WAIT_UNTIL_VALID_SECONDS exceeded, trying to submit anyway...<x1b[0m")

        # SUBMISSION
        tx_cbor = tx.to_cbor().hex()
        result = ogmios.query('SubmitTransaction', {'transaction': {'cbor': tx_cbor}})

        try:
            validity_interval_pos = (slot - validity_start)/(ttl - validity_start)
            time_histogram_node_position = min(7, max(-3, math.floor(validity_interval_pos*5)))+3
            time_histogram_node[time_histogram_node_position] += 1
            validity_interval_pos = (posix_to_slots(time.time()*1000.0, config.get('SHELLEY_OFFSET')) - validity_start)/(ttl - validity_start)
            time_histogram_now_position = min(7, max(-3, math.floor(validity_interval_pos*5)))+3
            time_histogram_now[time_histogram_now_position] += 1
        except:
            pass

        if 'result' in result and 'transaction' in result['result']:
            logger(f"<x1b[92mSUBMITTED block {out_block} => success! <x1b[0m {result['result']['transaction']['id']}")
            chain.add_submitted_tx([out_block, result['result']['transaction']['id'], TunaTxStatus.UNCONFIRMED])
            state[use_state] = None

            solution_submitted_timeout = time.time() + 90
            last_success_time = time.monotonic()

        elif 'error' in result and result['error']['code'] == 3117:
            chain.add_submitted_tx([out_block, None, TunaTxStatus.LATE])
            logger(f"<x1b[33mSUBMITTED<x1b[0m block {out_block}, but UTxOs already spent.")
            state[use_state] = None
        else:
            chain.add_submitted_tx([out_block, None, TunaTxStatus.ERROR])
            if validity_start >= slot or slot >= ttl:
                logger(f"<x1b[93mVALIDITY: outside of validity interval: {validity_start} < {slot} < {ttl}<x1b[0m")
            else:
                logger(f"VALIDITY: interval: {validity_start} < {slot} < {ttl}<x1b[0m")
            logger(f"<x1b[31mERROR: {result}<x1b[0m")

            try:
                if result['error']['code'] == 3125:
                    if 'insufficientlyFundedOutputs' in result['error']['data']:
                        ib = result['error']['data']['insufficientlyFundedOutputs']
                        if len(ib) == 1 and ib[0]['output']['address'] == CONTRACT_ADDRESS:
                            additional_lovelaces_needed = ib[0]['minimumRequiredValue']['ada']['lovelace'] - ib[0]['output']['value']['ada']['lovelace'] + 5000
                            if additional_lovelaces_needed < 100000:
                                additional_lovelaces_to_contract = additional_lovelaces_needed
                                chain.events['block'].set()
            except Exception as e:
                logger(f"<x1b[41mERROR: tried to recover from error, but failed for reason: {e}<x1b[0m")

        chain.check_submitted_tx_status()

        if config.get('DEBUG_TIMING'):
            try:
                logger(f"tx submission time distribution:")
                time_histogram_norm = min(1.0, 80.0/(max(sum(time_histogram_node),1)))
                logger( '-----+' +'-'*80)
                logger(f"     |<x1b[105m{' '*math.ceil(time_histogram_norm*time_histogram_node[0])}<x1b[0m")
                logger(f"     |<x1b[104m{' '*math.ceil(time_histogram_norm*time_histogram_node[1])}<x1b[0m")
                logger(f"     |<x1b[106m{' '*math.ceil(time_histogram_norm*time_histogram_node[2])}<x1b[0m")
                logger(f"V----|<x1b[102m{' '*math.ceil(time_histogram_norm*time_histogram_node[3])}<x1b[0m")
                logger(f"A    |<x1b[102m{' '*math.ceil(time_histogram_norm*time_histogram_node[4])}<x1b[0m")
                logger(f"L   >|<x1b[42m{' ' *math.ceil(time_histogram_norm*time_histogram_node[5])}<x1b[0m")
                logger(f"I    |<x1b[103m{' '*math.ceil(time_histogram_norm*time_histogram_node[6])}<x1b[0m")
                logger(f"D----|<x1b[43m{' ' *math.ceil(time_histogram_norm*time_histogram_node[7])}<x1b[0m")
                logger(f"     |<x1b[101m{' '*math.ceil(time_histogram_norm*time_histogram_node[8])}<x1b[0m")
                logger(f"     |<x1b[101m{' '*math.ceil(time_histogram_norm*time_histogram_node[9])}<x1b[0m")
                logger(f"     |<x1b[105m{' '*math.ceil(time_histogram_norm*time_histogram_node[10])}<x1b[0m")
                logger( '-----+' +'-'*80)
            except Exception as e:
                print(e)
