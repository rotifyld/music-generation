# based on https://stackoverflow.com/questions/27250954/transpose-block-of-text
def transpose_and_flip_upside_down_str(s: str) -> str:
    COLOR = '\033[93m'
    END = '\033[0m'

    rows = list([''.join(i) for i in zip(*s.split())])
    rows.reverse()
    colored_rows = [''.join([COLOR + row[48 * i] + END + row[48 * i + 1:48 * (i + 1)] for i in range(16)])
                    for row in rows]
    return '\n'.join(colored_rows)

# 96 144
