class bcolors:
    HEADER = '\033[95m'
    INFO = '\033[94m'
    OK = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


_PREFIX = '>>> '


def log_ok(s: str):
    print(bcolors.OK + _PREFIX + s + bcolors.ENDC)


def log_info(s: str):
    print(bcolors.INFO + _PREFIX + s + bcolors.ENDC)


def log_warning(s: str):
    print(bcolors.WARNING + _PREFIX + s + bcolors.ENDC)


def log_error(s: str):
    print(bcolors.ERROR + _PREFIX + s + bcolors.ENDC)
