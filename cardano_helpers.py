from contextlib import contextmanager
import sys, os
import pycardano
import datetime
import json

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def get_assets_with_policy(output, policy_id_hex):
    assets = output.amount.multi_asset.get(pycardano.hash.ScriptHash(bytes.fromhex(policy_id_hex)))
    return {k.to_primitive().hex(): v for k,v in assets.items()} if assets else {}

def get_asset_amount(output, policy_id_hex, assetname_hex):
    return output.amount.multi_asset[pycardano.hash.ScriptHash(bytes.fromhex(policy_id_hex))][pycardano.AssetName(bytes.fromhex(assetname_hex))]

def format_timedelta_since_posix_time(pt):
    date_now = datetime.datetime.now()
    date_pt = datetime.datetime.fromtimestamp(pt * 0.001)
    if date_pt > date_now:
        return ""
    date_diff = date_now - date_pt

    days = date_diff.days
    years = days // 365
    if years > 0:
        return f">{years} years"
    months = days // 30
    if months > 0:
        return f">{months} months"
    if days > 0:
        return f">{days} days"
    hours = date_diff.seconds // 3600
    if hours > 0:
        return f"{hours} hours"
    minutes = date_diff.seconds // 60
    if minutes > 0:
        return f"{minutes} minutes"
    return f"{date_diff.seconds} seconds"

def format_seconds(s):
    if s > 86400*7:
        return f">{s/(86400*7):.0f} weeks"
    elif s > 86400:
        return f">{s/(86400):.0f} days"
    elif s > 3600:
        return f"{s/(3600):.1f} hours"
    elif s > 180:
        return f"{s/(60):.0f} minutes"
    elif s > 60:
        return f"{s/(60):.1f} minutes"
    elif s > 5:
        return f"{s:.0f} seconds"
    elif s > 1:
        return f"{s:.1f} seconds"
    else:
        return "<1 second"


def posix_to_slots(pt, shelley_offset=1666656000):
    return (pt//1000) - shelley_offset

def slots_to_posix(slots, shelley_offset=1666656000):
    return (slots + shelley_offset)*1000

def get_difficulty_adjustment(total_epoch_time, epoch_target):
    if epoch_target / total_epoch_time >= 4 and epoch_target % total_epoch_time > 0:
        return 1, 4
    elif total_epoch_time / epoch_target >= 4 and total_epoch_time % epoch_target > 0:
        return 4, 1
    else:
        return total_epoch_time, epoch_target

def get_new_difficulty(difficulty_number, current_leading_zeros, adjustment_numerator, adjustment_denominator, padding=16):
    new_padded_difficulty = difficulty_number * padding * adjustment_numerator // adjustment_denominator
    new_difficulty = new_padded_difficulty // padding

    if new_padded_difficulty // 65536 == 0:
        if current_leading_zeros >= 60:
            return 4096, 60
        else:
            return new_padded_difficulty, current_leading_zeros + 1
    elif new_difficulty // 65536 > 0:
        if current_leading_zeros <= 2:
            return 65535, 2
        else:
            return new_difficulty // padding, current_leading_zeros - 1
    else:
        return new_difficulty, current_leading_zeros

def calc_diff(b):
    lz = 0
    dn = 0
    i = 0
    while i < len(b):
        if b[i] == 0:
            lz += 2
        elif b[i] < 16:
            lz += 1
            dn = b[i]<<12
            if i < len(b)-1:
                dn += b[i+1]<<4
            if i < len(b)-2:
                dn += b[i+1]>>4
            break
        else:
            dn = b[i]<<8
            if i < len(b)-1:
                dn += b[i+1]
            break
        i += 1
    return lz, dn

def tx_input_from_utxo(utxo):
    if utxo is None:
        return None
    return pycardano.transaction.TransactionInput(pycardano.hash.TransactionId(bytes.fromhex(utxo['transaction']['id'])), utxo['index'])

def value_from_utxo(utxo):
    value = utxo['value']
    multi_asset_dict = {}
    for policy_id, policy_assets in utxo['value'].items():
        if policy_id == 'ada':
            continue
        policy_dict = {}
        for asset,amount in policy_assets.items():
            policy_dict[bytes.fromhex(asset)] = amount
        multi_asset_dict[bytes.fromhex(policy_id)] = policy_dict
    coin = utxo['value']['ada']['lovelace']
    return pycardano.transaction.Value.from_primitive([coin, multi_asset_dict])


def get_collateral_utxo(utxos):
    for utxo in utxos:
        if 'ada' in utxo['value'] and len(utxo['value'].keys()) == 1 and utxo['value']['ada']['lovelace'] > 7000000 and utxo['value']['ada']['lovelace'] < 30000000:
            return utxo
    for utxo in utxos:
        if 'ada' in utxo['value'] and utxo['value']['ada']['lovelace'] > 7000000:
            return utxo
    return None

def get_tx_input_utxo(utxos):
    for utxo in utxos:
        if 'ada' in utxo['value'] and utxo['value']['ada']['lovelace'] > 2500000:
            return utxo
    return None

def get_nft_utxo(utxos, policy_hex, name_hex):
    for utxo in utxos:
        if utxo['value'].get(policy_hex, {}).get(name_hex, 0) == 1:
            return utxo
    return None

def unique_utxos(utxos):
    return list(json.loads(y) for y in set(json.dumps(x) for x in utxos))

def lovelace_value_from_utxos(utxos):
    return sum(utxo.get('value', {}).get('ada', {}).get('lovelace', 0) for utxo in unique_utxos(utxos))

def token_value_from_utxos(utxos, policy, assetname_hex):
    return sum(utxo.get('value', {}).get(policy, {}).get(assetname_hex, 0) for utxo in unique_utxos(utxos))

