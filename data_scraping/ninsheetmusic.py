from logger import log_error, log_ok

from joblib import Parallel, delayed
import requests


def download_single(i: int):
    try:
        url = 'https://www.ninsheetmusic.org/download/mid/{}'.format(i)
        r = requests.get(url, allow_redirects=True)
        open('ninsheetmusic/{}.mid'.format(i), 'wb').write(r.content)
        log_ok('Success at {}'.format(i))
    except Exception as e:
        log_error('Error at {}: {}'.format(i, str(e)))


results = Parallel(n_jobs=-1, verbose=2, backend="threading")(map(delayed(download_single), range(4400)))
