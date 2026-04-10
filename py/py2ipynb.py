#!/usr/bin/env python3

import sys
import json

s = sys.stdin.read()

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
            return dict(
                cell_type="markdown",
                metadata=dict(trusted=True),
                source=[
                    "\n".join(((x.removeprefix("# ") if x != "#" else "") for x in source.splitlines())),
                ],
            )
        case _:
            assert False, f"invalid cell_type {ct}, {source[:min(60, len(source))]}"



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
        cells=[parse_cell(cell) for cell in s.split("\n###")],
    ),
    sys.stdout,
)

