#!/usr/bin/bash
cd "${0%/*}" || exit                                # Run from this directory
#------------------------------------------------------------------------------
tm=$(awk '/ExecutionTime/{a=$3} END{print(a)}' log.reactingFoam)
if [ -z "$tm" ]; then
    echo "0"
else
    echo "$tm"
fi
#------------------------------------------------------------------------------
