#!/usr/bin/env python3

import sys
import json

def parse_cell(s):
    ct, source = s.removeprefix("###").split("\n", maxsplit=1)

    match ct:
        case "":
            return dict(
                cell_type="code",
                metadata=dict(trusted=True),
                source=[source.strip()],
            )
        case "md":
            lines = [
                (x.removeprefix("# ") if x != "#" else "")
                for x in source.strip().splitlines()
            ]

            return dict(
                cell_type="markdown",
                metadata=dict(trusted=True),
                source=["\n".join(lines)],
            )
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
    json.dump(
        dict(
            metadata= dict(
                kernelspec=dict(
                    display_name="Python 3 (ipykernel)",
                    language="python",
                    name="python3",
                ),
            ),
            nbformat=4,
            nbformat_minor=0,
            cells=[parse_cell(cell) for cell in fin.read().split("\n###")],
        ),
        fout,
        indent=2,
    )

