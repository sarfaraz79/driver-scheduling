#!/usr/bin/env bash

exec jq -r '.cells[] | select(.cell_type == "code") | .source | add | add(., "\n\n###\n")'
