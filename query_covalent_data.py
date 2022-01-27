import requests
import pickle
import argparse
import logging
from covalent_api import *

from config import API_KEY


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# get debug messages from the covalent_api module
# this just lets us see how individual pages are being queried
logging.getLogger('covalent_api.covalent_api').setLevel(logging.DEBUG)


def main(pool_info, output_file):
    pool_info = pd.read_csv(pool_info)
    logger.info('Loaded pool info data.')

    data = {}
    logger.info('Beginning data queries.')
    for i, row in pool_info.iterrows():
        token_0 = row['contract_ticker_symbol_token_0']
        token_1 = row['contract_ticker_symbol_token_1']
        pool_fee = row['pool_fee']
        pool_id = f'{token_0}-{token_1}-{pool_fee}'

        try:
            swaps = get_uniswapv3_swaps(
                api_key=API_KEY,
                pool_address=row['pool_contract_address'],
                page_size=5000,
                total_pages=None
            )
            logger.info(f'Queried swap data for {pool_id}.')
        except (APIError, requests.ConnectionError) as e:
            logger.warning(
                f'Error querying swap data for {pool_id}.\n'
                f'{type(e).__name__}: {e}'
            )
            swaps = None

        try:
            liquidity = get_uniswapv3_liquidity(
                api_key=API_KEY,
                pool_address='0x60594a405d53811d3bc4766596efd80fd545a270',
                page_size=5000,
                total_pages=None
            )
            logger.info(f'Queried liquidity data for {pool_id}.')
        except (APIError, requests.ConnectionError) as e:
            logger.warning(
                f'Error querying liquidity data for {pool_id}.\n'
                f'{type(e).__name__}: {e}'
            )
            liquidity = None

        data[pool_id] = {'swaps': swaps, 'liquidity': liquidity}

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
        logger.info(f'Dataset saved to {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pool_info', required=True,
                        help='pool_info.csv file with the required pool data.')
    parser.add_argument('-o', '--output_file', required=True,
                        help='File path to save the resulting data.')
    args = parser.parse_args()

    main(args.pool_info, args.output_file)
