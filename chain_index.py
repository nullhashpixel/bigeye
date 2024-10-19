# -*- coding: utf-8 -*-
import os
import sys
import sqlite3
import threading
import json
import time

try:
    from flask import Flask
    from flask import request
except:
    pass

class ChainIndex:
    def __init__(self, profile, watcher=None):
        self.profile = profile
        self.config = profile.config
        self.dir = profile.profile_dir
        self.db_filename = self.dir + '/store.db'
        self.init_dir()
        self.watcher = watcher

        self.con = sqlite3.connect(self.db_filename)
        self.con.row_factory = sqlite3.Row
        self.cur = self.con.cursor()

        self.reset()

        if self.config.get('CHAIN_STATE_WEBSERVER'):
            app = Flask(__name__)

            db_filename = self.db_filename

            @app.route("/tip")
            def api_tip():
                t0 = time.time()
                db = sqlite3.connect(db_filename)
                db.row_factory = sqlite3.Row
                cur = db.cursor()

                result = {}
                try:
                    cur.execute("SELECT block, slot, id, tx, tuna_block, tuna_merkle_root FROM chain ORDER BY tuna_block DESC LIMIT 1;")
                    row = cur.fetchone()
                    result['result'] = dict(row)
                    db.close()
                except Exception as e:
                    print(e)
                    result['error'] = 'error'
                print(f"took {time.time()-t0} seconds")
                return json.dumps(result)

            @app.route("/block/<block>")
            def api_block(block):
                try:
                    block_number = int(block)
                except:
                    return json.dumps({'error': 'block number must be an integer.'})

                t0 = time.time()
                db = sqlite3.connect(db_filename)
                db.row_factory = sqlite3.Row
                cur = db.cursor()

                result = {}
                try:
                    cur.execute("SELECT tuna_block, tuna_epoch, tuna_lz, tuna_dn, tuna_hash, tuna_nonce, tuna_miner, tuna_miner_cred_hash FROM chain WHERE tuna_block = ?;", (block_number,))
                    row = cur.fetchone()
                    result['result'] = dict(row)
                    db.close()
                except Exception as e:
                    print(e)
                    result['error'] = 'error'
                print(f"took {time.time()-t0} seconds")
                return json.dumps(result)

            @app.route("/blocks")
            def api_blocks():
                result = {}
                db = sqlite3.connect(db_filename)
                db.row_factory = sqlite3.Row
                cur = db.cursor()
                try:
                    start_block = int(request.args.get('s') or -1)
                    direction = int(request.args.get('d') or 0)
                    columns = "block,slot,id,tx,tuna_block,tuna_hash,tuna_lz,tuna_dn,tuna_epoch,tuna_posix_time,tuna_merkle_root,tuna_miner,tuna_miner_cred_hash,tuna_miner_data"
                    if direction == 1:
                        if start_block < 1:
                            cur.execute(f"SELECT {columns} FROM chain ORDER BY tuna_block ASC LIMIT 10;")
                        else:
                            cur.execute(f"SELECT {columns} FROM chain WHERE tuna_block > :tuna_block ORDER BY tuna_block ASC LIMIT 10;", {"tuna_block": start_block})
                    else:
                        if start_block < 1:
                            cur.execute(f"SELECT {columns} FROM chain ORDER BY tuna_block DESC LIMIT 10;")
                        else:
                            cur.execute(f"SELECT {columns} FROM chain WHERE tuna_block < :tuna_block ORDER BY tuna_block DESC LIMIT 10;", {"tuna_block": start_block})
                    result['result'] = [dict(x) for x in cur.fetchall()]
                except Exception as e:
                    print(e)
                    result['error'] = 'error'
                db.close()
                return result

            @app.route("/proof/<data>")
            def api_proof(data):
                try:
                    hash_bytes = bytes.fromhex(data)
                except:
                    return json.dumps({'error': 'invalid hex data'})

                result = {}
                t0 = time.time()
                try:
                    t = self.watcher.state['tuna'].state['trie']
                    result['result'] = {'cbor': t.prove_digest(hash_bytes).toCBOR()}
                except Exception as e:
                    print(e)
                    result['error'] = 'proof error'
                print(f"took {time.time()-t0} seconds")
                return json.dumps(result)


            if ':' in self.config.get('CHAIN_STATE_WEBSERVER'):
                host_name, port = self.config.get('CHAIN_STATE_WEBSERVER').split(':')
            else:
                host_name, port = "127.0.0.1", 61631
            threading.Thread(target=lambda: app.run(host=host_name, port=int(port), debug=True, use_reloader=False)).start()

    def init_dir(self):
        try:
            os.makedirs(self.dir)
        except:
            pass

    def reset(self):
        self.cardano_state = {}
        self.tuna_state = {}

        self.cur.execute("""CREATE TABLE IF NOT EXISTS chain (
                block UNSIGNED BIG INT,
                slot UNSIGNED BIG INT,
                id VARCHAR(64),
                tx VARCHAR(64),
                tuna_block UNSIGNED BIG INT,
                tuna_hash VARCHAR(64),
                tuna_lz INT,
                tuna_dn INT,
                tuna_epoch BIGINT,
                tuna_posix_time BIGINT,
                tuna_merkle_root VARCHAR(64)
            );""")
        self.cur.execute("""CREATE TABLE IF NOT EXISTS submissions (
                tuna_block UNSIGNED BIG INT,
                tuna_hash VARCHAR(64),
                confirmed INT,
                submit_time UNSIGNED BIG INT
                );""")

        try:
            self.cur.execute('ALTER TABLE chain ADD COLUMN cbor BLOB;')
        except:
            pass

        try:
            self.cur.execute('ALTER TABLE chain ADD COLUMN tuna_miner VARCHAR(56);')
        except:
            pass

        try:
            self.cur.execute('ALTER TABLE chain ADD COLUMN tuna_nonce TEXT;')
        except:
            pass

        try:
            self.cur.execute('ALTER TABLE chain ADD COLUMN tuna_miner_cred_hash TEXT;')
        except:
            pass

        try:
            self.cur.execute('ALTER TABLE chain ADD COLUMN tuna_miner_data TEXT;')
        except:
            pass

    def get_tuna_block(self, tuna_block):
        self.cur.execute("SELECT block FROM chain WHERE tuna_block = ?", (tuna_block,))
        results = self.cur.fetchone()
        return results[0] if (results is not None and len(results) > 0) else None

    def get_chain(self):
        self.cur.execute("SELECT tuna_block, tuna_hash, tuna_merkle_root FROM chain ORDER BY tuna_block;")
        return self.cur.fetchall()

    def insert(self, record):
        existing_tuna_block = self.get_tuna_block(record['tuna_block'])
        if existing_tuna_block:
            print("TODO: handle rollbacks")
            os._exit(17)

        self.cur.execute("""INSERT INTO chain (block, slot, id, tx, tuna_block, tuna_hash, tuna_lz, tuna_dn, tuna_epoch, tuna_posix_time, tuna_merkle_root, cbor, tuna_miner, tuna_nonce, tuna_miner_cred_hash, tuna_miner_data)
                            VALUES (:block, :slot, :id, :tx, :tuna_block, :tuna_hash, :tuna_lz, :tuna_dn, :tuna_epoch, :tuna_posix_time, :tuna_merkle_root, :cbor, :tuna_miner, :tuna_nonce, :tuna_miner_cred_hash, :tuna_miner_data);""", record);
        self.con.commit()

    def __repr__(self):
        return f"<ChainIndex:{self.db_filename}>"

    def get_state(self):
        self.cur.execute("SELECT * FROM chain ORDER BY tuna_block DESC LIMIT 1;")
        result = self.cur.fetchone()
        return dict(result) if result else None 

    def rollback(self, height):
        self.cur.execute("DELETE FROM chain WHERE block > :height;", {'height': height})
        self.con.commit()
        return self.cur.rowcount

