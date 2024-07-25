import asyncio

from starknet_py.contract import Contract
from starknet_py.net.full_node_client import FullNodeClient
from starknet_py.net.account.account import Account
from starknet_py.net.signer.stark_curve_signer import KeyPair

NODE_URL = "http://127.0.0.1:5050"
CONTRACT_ADDRESS = "0xf62381a34a379b8c7468865cb8874ef0ae18c1833b55c7f04a88f472912fe5"

async def main():
    client = FullNodeClient(node_url=NODE_URL)

    # setup account
    account = Account(
        client=client,
        address="0x34ba56f92265f0868c57d3fe72ecab144fc96f97954bbbc4252cef8e8a979ba",
        key_pair=KeyPair(private_key=int("0xb137668388dbe9acdfa3bc734cc2c469", 16), public_key=int(
            "0x5a5e37c60e77a0318643b111f88413a76af6233c891a0cfb2804106372006d4", 16)),
        chain=0x534e5f5345504f4c4941
    )

    # connect to contract
    contract = await Contract.from_address(provider=account, address=CONTRACT_ADDRESS)

    # call contract
    result = await contract.functions["get_weights"].call(
        [1,9000], 0
    )

    print(result)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
