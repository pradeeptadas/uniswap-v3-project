import requests
import argparse
import pickle
import time
import logging

from covalent_api import *
from config import API_KEY


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# get debug messages from the covalent_api module
# this just lets us see how individual pages are being queried
logging.getLogger('covalent_api').setLevel(logging.DEBUG)


def main(raw_data, output_file):
    with open(raw_data, 'rb') as f:
        data = pickle.load(f)

    all_data = {}
    for pool, dset in data.items():
        removes = dset['liquidity'].loc[dset['liquidity']['liquidity_event'] == 'REMOVE_LIQUIDITY', :]
        tx_hashes = removes['tx_hash'].unique()
        pool_data = {}

        for tx_hash in tx_hashes:
            while True:
                try:
                    tx_data = get_transaction(api_key=API_KEY, tx_hash=tx_hash,
                                              n_attempts=1, retry_codes=tuple())
                    break
                except (APIError, requests.exceptions.Timeout) as e:
                    logger.warning(f"{type(e).__name__}: {e}.")
                    time.sleep(10)

            pool_data[tx_hash] = tx_data

        all_data[pool] = pool_data
        logger.info(f'Completed pool {pool}.')

    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--raw_data', required=True,
                        help='pool_data_raw.csv file.')
    parser.add_argument('-o', '--output_file', required=True,
                        help='File path to save the resulting data.')
    args = parser.parse_args()

    main(args.raw_data, args.output_file)
