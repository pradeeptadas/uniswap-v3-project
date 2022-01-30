import requests
import pandas as pd
import numpy as np
import itertools
import logging


logger = logging.getLogger(__name__)
BASE_URL = 'https://api.covalenthq.com'


class APIError(Exception):
    pass


def unravel_token(df, token_col):
    token_df = df[token_col].apply(pd.Series)
    token_df.rename(columns={
        col: f"{col}_{token_col}" for col in token_df.columns
    }, inplace=True)

    return pd.concat([df.drop([token_col], axis=1), token_df], axis=1)


# mostly used to handle the 507 error 
# as all we can do is make the request again and hope it gets in the queue
def try_n_gets(n=2, retry_codes=(507,), get_args=[], get_kwargs={}):
    assert n >= 1, 'n must be greater than or equal to 1.'
    for _ in range(n):
        response = requests.get(*get_args, **get_kwargs)
        if response.status_code in retry_codes:
            continue
        else:
            return response
    # only executes if the for loop above does not break/return
    else:
        return response


def raise_api_error(response):
    try:
        content = response.json()
        msg = f"{content['error_code']} {content['error_message']}"
    except ValueError:  # JSONDecodeError inherits from ValueError
        msg = f"{response.status_code} {response.reason}"

    raise APIError(msg)


def get_covalent_data(api_key, endpoint, params=None,
                      page_size=100, page_number=None,
                      n_attempts=1, retry_codes=(507,)):
    url = BASE_URL + endpoint
    if params is None:
        params = {'page-size': page_size}
    else:
        params['page-size'] = page_size

    i = 0 if page_number is None else page_number
    all_responses = []
    has_more = True
    while has_more:
        params['page-number'] = i
        args = [url]
        kwargs = {'auth': (api_key, ''), 'params': params}
        response = try_n_gets(n=n_attempts, retry_codes=retry_codes,
                              get_args=args, get_kwargs=kwargs)

        if response.ok:
            content = response.json()
            logger.debug(f'Queried page {i} for endpoint: {endpoint}')
            all_responses.append(response)
            has_more = (
                False if content['data']['pagination'] is None
                else content['data']['pagination']['has_more']
            )
            i += 1
        elif response.status_code in retry_codes:  # happens when all attempts fail
            logger.warning(
                f"Could not request page {i} after {n_attempts} attempts. "
                f"Page {i} was skipped."
            )
        else:
            raise_api_error(response)

        if page_number is not None:
            break

    all_content = [r.json()['data']['items'] for r in all_responses]
    all_content = list(itertools.chain(*all_content))

    return all_content


def get_uniswapv3_pools(api_key=None, page_size=1000,
                        n_attempts=10, retry_codes=(507,)):
    # chain_id=1 as Uniswap v3 is on the Ethereum Mainnet
    endpoint = '/v1/1/uniswap_v3/pools/'
    all_content = get_covalent_data(api_key, endpoint,
                                    page_size=page_size,
                                    n_attempts=n_attempts,
                                    retry_codes=retry_codes)

    df = pd.DataFrame(all_content)
    if df.shape[0] > 0:
        df = unravel_token(df, 'token_0')
        df = unravel_token(df, 'token_1')

    return df


def get_uniswapv3_liquidity(api_key=None, pool_address=None,
                            page_size=1000, page_number=None,
                            n_attempts=10, retry_codes=(507,)):
    # chain_id=1 as Uniswap v3 is on the Ethereum Mainnet 
    endpoint = f"/v1/1/uniswap_v3/liquidity/address/{pool_address}/"
    all_content = get_covalent_data(api_key, endpoint,
                                    page_size=page_size,
                                    page_number=page_number,
                                    n_attempts=n_attempts,
                                    retry_codes=retry_codes)

    df = pd.DataFrame(all_content)
    if df.shape[0] > 0:
        df = unravel_token(df, 'token_0')
        df = unravel_token(df, 'token_1')

    return df


def get_uniswapv3_swaps(api_key=None, pool_address=None,
                        page_size=1000, page_number=None,
                        n_attempts=10, retry_codes=(507,)):
    # chain_id=1 as Uniswap v3 is on the Ethereum Mainnet 
    endpoint = f"/v1/1/uniswap_v3/swaps/address/{pool_address}/"
    all_content = get_covalent_data(api_key, endpoint,
                                    page_size=page_size,
                                    page_number=page_number,
                                    n_attempts=n_attempts,
                                    retry_codes=retry_codes)

    df = pd.DataFrame(all_content)
    if df.shape[0] > 0:
        df = unravel_token(df, 'token_0')
        df = unravel_token(df, 'token_1')

    return df


def get_uniswapv3_counts(api_key=None, pool_address=None, transaction='swaps',
                         n_attempts=10, retry_codes=(507,)):
    # chain_id=1 as Uniswap v3 is on the Ethereum Mainnet 
    if transaction == 'swaps':
        endpoint = f"/v1/1/uniswap_v3/swaps/address/{pool_address}/"
    elif transaction == 'liquidity':
        endpoint = f"/v1/1/uniswap_v3/liquidity/address/{pool_address}/"
    else:
        raise ValueError(f"{transaction} is not a valid transaction type.")

    url = BASE_URL + endpoint
    params = {'page-size': 1, 'page-number': 0}

    args = [url]
    kwargs = {'auth': (api_key, ''), 'params': params}
    response = try_n_gets(n=n_attempts, retry_codes=retry_codes,
                          get_args=args, get_kwargs=kwargs)

    if response.ok:
        content = response.json()
        return content['data']['pagination']['total_count']
    elif response.status_code in retry_codes:  # happens when all attempts fail
        logger.warning(
            f"Could not request page count data after {n_attempts} attempts. "
            f"None was returned."
        )
        return None
    else:
        raise_api_error(response)
