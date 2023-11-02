#!/usr/bin/bash
cd "${0%/*}" || exit                                # Run from this directory
#------------------------------------------------------------------------------
tm=$(awk 'function abs(v) {return v < 0 ? -v : v} /continuity errors/{a=$15} END{print(abs(a))}' log.reactingFoam)
if [ -z "$tm" ]; then
    echo "500000"
else
    echo "$tm"
fi
#------------------------------------------------------------------------------
