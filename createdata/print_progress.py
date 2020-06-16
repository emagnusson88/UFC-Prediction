import sys
from datetime import datetime

def print_progress(iteration: int, total: int, prefix: str='', suffix: str='',
                    decimals: int=1, bar_length: int =50) -> None:
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    startTime = datetime.now()

    percents = f'{100 * (iteration / float(total)):.2f}'
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = f'{"█" * filled_length}{"-" * (bar_length - filled_length)}'

    sys.stdout.write(f'\r{prefix} |{bar}| {percents}% {suffix}')

    if iteration == total:
        sys.stdout.write('\n')
        print(datetime.now() - startTime)

    sys.stdout.flush()
