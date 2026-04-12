#!/usr/bin/env python3

import sys
import json

def parse_cell(cell):
    match cell["cell_type"]:
        case "code":
            return f"###\n\n{"".join(cell["source"]).strip()}\n\n"
        case "markdown":
            lines = [
                (f"# {x}" if (x := line.rstrip()) != "" else "#")
                for line in "".join(cell["source"]).strip().splitlines()
            ]

            return f"###md\n\n{"\n".join(lines)}\n\n"
        case _:
            assert False, f"invalid cell_type {ct}, {source[:min(60, len(source))]}"

fin = sys.stdin
fout = sys.stdout

match sys.argv[1:]:
    case []:
        pass

    case [pin]:
        fin = open(pin)

    case [pin, pout]:
        fin = open(pin)
        fout = open(pout, mode="w")

    case _:
        assert False, "invalid args"

with fin, fout:
    nb = json.load(fin)
    for cell in nb["cells"]:
        fout.write(parse_cell(cell))

