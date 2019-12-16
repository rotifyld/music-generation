# based on https://stackoverflow.com/questions/27250954/transpose-block-of-text
def transpose_and_flip_upside_down_str(measures: int, atoms: int, s: str) -> str:
    COLOR = '\033[93m'
    END = '\033[0m'

    rows = list([''.join(i) for i in zip(*s.split())])
    rows.reverse()
    colored_rows = [
        ''.join([COLOR + row[atoms * i] + END + row[atoms * i + 1:atoms * (i + 1)] for i in range(measures)])
        for row in rows]
    return '\n'.join(colored_rows)
