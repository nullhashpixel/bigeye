# bigeye miner

requirements:
- Python >=3.8
- **Ogmios 6.6.*, connected to a Cardano node**
- Ogmios needs **CBOR OUTPUT ENABLED**

tested with local Ogmios/node, demeter should also work

## setup

### install requirements

clone code
````
git clone https://github.com/nullhashpixel/bigeye.git
cd bigeye
````

Recommendation: install requirements in a virtual requirement.

````
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

### build miner cores

This step needs the standard gcc C++ build tools, on Ubuntu it can be installed with
````
apt update
apt install build-essential
````
#### CPU

go to `miners/cpu` and run
````
make
````
#### GPU

Use `clinfo` to check if OpenCL is already installed.
If OpenCL is not installed, run (for Nvidia GPUs):
````
apt update
apt install opencl-headers clinfo nvidia-opencl-dev
````

go to `miners/gpu` and run
````
make
````

Test if the GPU miner works with:
````
./cltuna benchmark
````
It will compute a few hashes and output a hash rate.

If you have a Nvidia GPU and the miner is running, you can check if it is properly using the GPU with
````
nvidia-smi
````
GPU-Util should be >=99% and you should see `./cltuna` in the "Processes:" list.
It improves the performance if other applications using the GPU are closed while mining, check the "Processes:" list for other programs using the GPU.


### wallet

- create a new wallet and put the seed phrase in a file named `wallet.txt` in `config/mainnet/` (or similarly for other profiles)
- fund the wallet with a few (t)ADA (at least 10 (t)ADA are needed, a single UTxO is enough)
- for security reasons don't re-use existing wallets and only keep little amounts in the wallet used to mine

### configuration

- run the miner, upon first launch it will ask for missing config values (e.g. Ogmios URL)
- or change them manually in `config/mainnet/config.json`

## profiles

profiles are used to organize mining for different versions of $TUNA (possible future forks), mine with different settings or mine on one of the testnets.
The default profile is called `mainnet`.

The **config file** for this profile is located at `config/mainnet/config.json`.
If the `config.json` file does not exist, it will be created during the first start of the miner.


## important note

bigeye runs transaction building and hash computation in different processes to let users run them on different machines.
Communication between is via low-level TCP sockets and requires open ports on the machine running the hash computations.
The machine running the transaction building (holding private keys for the wallet etc.) does not need to be accessible from the outside and can also run behind a NAT (router, firewall).

In a simple setup, transaction building and hash computation can run on the same machine.
The `config.json` offers a simple way to start the miner cores (which perform hash computation) automatically with the
````
    "AUTO_SPAWN_MINERS": true,
````
setting.
````
    "MINER_EXECUTABLE": "miners/cpu/cpu-sha256",
    "MINERS": "127.0.0.1:2023-2034",
````
define which executable to spawn and the range `2023-2034` specifies 12 processes to be spawned, which each will listen on the ports in this range.

If the miner cores run on other machines, an example configuration could look like this:
````
    "AUTO_SPAWN_MINERS": false,
    "MINERS": "192.168.0.100:2023,192.168.0.100:2024,192.168.0.101:2023",
````

The `simple` miner core is a pure Python implementation, which should be much slower on most systems.


## mine
````
./mine.py mainnet
````

To auto-restart the miner after a potential crash, create a script `run.sh`

```` bash
#!/bin/bash
while true; do
./mine.py mainnet;
echo "waiting 10 sec";
sleep 10;
done
````

make it executable
````
chmod +x run.sh
````

and then use
````
./run.sh
````
to mine.

## sample GPU configuration

````
...
    "AUTO_SPAWN_MINERS": true,
    "MINER_EXECUTABLE": "miners/gpu/run.sh",
    "MINERS": "127.0.0.1:2023",
...
````
# using a demeter ogmios instance:

this disables SSL certificate validation, only use if you know what you're doing!

````
    "OGMIOS": "wss://dmtr_ogmios1{rest_of_your_api_key}.mainnet-v6.ogmios-m1.demeter.run",
    "OGMIOS_DISABLE_CERTIFICATE_CHECK": true,
````

# docker

minimum setup to run (not compile!) the GPU miner in a docker (on GPU renting platforms)

image: `ubuntu:20.04` (official Ubuntu 20)
````
apt install -y clinfo nvidia-opencl-dev
````

warning: when running the miner on cloud services use only trusted images (wallet stealers are common on some platforms where users can share docker images) and only run the miner core (`cltuna`) in the cloud, not the tx building part! Never copy your seed phrase to untrusted cloud machines.

# hash rates

| GPU     | OS     | hash rate GH/s   |
|---------|--------|------------------|
| 4090    | Linux  | 4.3              |
| 4080S   | Linux  | 2.2              |
| 4070    | Linux  | 1.8              |
| 3080    | Linux  | 1.7              |
| 3070    | Linux  | 1.1              |
| 2080S   | Linux  | 1.1              |
| A4000   | Linux  | 1.0              |
| 4060    | Linux  | 0.95             |

performance can vary depending on the exact GPU and PC

# vast.ai template:

Use these template settings to create your own template to mine. (be careful with existing templates, they might contain malware!)


````
Image Path/Tag: ubuntu:20.04
Version Tag: 20.04
Docker Options: -p 2023:2023
Run interactive shell servers, SSH: yes
Use direct SSH connection...: yes
On-Start Script: mkdir /workspace && apt install -y clinfo nvidia-opencl-dev
````

deploy.sh
````
#!/bin/bash
scp -o "StrictHostKeyChecking accept-new" -i {YOUR_SSH_IDENTITY_FILE} -P $2 sha256_opencl.cl cltuna run.sh root@${1}:/workspace
ssh root@${1} -p ${2} -i {YOUR_SSH_IDENTITY_FILE} -t << EOF
   tmux kill-session -a
   tmux new-session -t session1 -d
   tmux list-sessions
   tmux send-keys -t session1-0 "cd /workspace; ./run.sh 2023" ENTER
EOF
````

use (from miners/gpu directory):
````
./deploy.sh 12.34.56.78 45678
````
important:
 - the port to be used as argument to `deploy.sh` is the one shown in "IP & Port Info" redirecting to "22/tcp".
 - the port added to the config.json is the port redirecting to "2023/tcp"



# Known bugs/limitations

- It should work on Windows with some extra steps, as outlined here: https://github.com/nullhashpixel/cltunaminer
- It doesn't properly handle all possible error conditions of Ogmios/cardano-node. For robustness, run the miner in the `run.sh` script as shown above and add a timeout for submitting a valid transaction with `"GLOBAL_TIMEOUT": 3600` in the `config.json`. This will exit the miner after 3600 s without a successfully submitted transaction, after which it will be restarted.


# Disclaimer

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.






