import pycardano
from hashlib import blake2b
from pycardano.plutus import CBORTag
from pycardano.plutus import RawPlutusData
from pycardano.serialization import IndefiniteList
from pycardano.serialization import ByteString

class TargetState:
    def __init__(self, wallet_pkh, VERSION_INFO, nft=None):
        self.wallet_pkh = wallet_pkh
        self.VERSION_INFO = VERSION_INFO
        self.nonce = bytes([0]*16)
        self.miner = bytes([0]*32)
        self.epoch_time = 123
        self.block_number = 1234
        self.current_hash = bytes([0]*32)
        self.leading_zeros = 7
        self.target_number = 16384
        self.nft = nft
        self.miner_credential = None
        self.proof = None

        self._use_preview_v1_serialization = False

    def serialize(self):
        VERSION_INFO_bytes = self.VERSION_INFO.encode('utf-8') if type(self.VERSION_INFO) is str else self.VERSION_INFO

        if self.nft is None:
            miner_cred = RawPlutusData.from_primitive(
                    CBORTag(121, [
                        self.wallet_pkh,
                        VERSION_INFO_bytes
                        ]))
            miner_cred_hash = blake2b(miner_cred.to_cbor(), digest_size=32).digest()
            self.miner_credential = miner_cred
        else:
            extra_data = VERSION_INFO_bytes
            if 'extra' in self.nft:
                extra_data += (' ' + self.nft.get('extra', '')).encode('utf-8')
            miner_cred = RawPlutusData.from_primitive(
                    CBORTag(122, [
                        bytes.fromhex(self.nft['policy']),
                        bytes.fromhex(self.nft['name']),
                        self.nft.get('output_index', 0),
                        extra_data
                        ]))
            miner_cred_hash = blake2b(miner_cred.to_cbor(), digest_size=32).digest()
            self.miner_credential = miner_cred

        if self._use_preview_v1_serialization:
            res = RawPlutusData.from_primitive(
                    CBORTag(121, [
                        self.nonce,
                        miner_cred_hash,
                        self.epoch_time,
                        self.block_number,
                        self.current_hash,
                        self.leading_zeros,
                        self.target_number
                        ]))
        else:
            res = RawPlutusData.from_primitive(
                    CBORTag(121, [
                        self.nonce,
                        miner_cred_hash,
                        self.block_number,
                        self.current_hash,
                        self.leading_zeros,
                        self.target_number,
                        self.epoch_time
                        ]))
        return res.to_cbor()

