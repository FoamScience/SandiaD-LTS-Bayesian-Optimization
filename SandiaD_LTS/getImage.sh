#!/usr/bin/bash
cd "${0%/*}" || exit                                # Run from this directory
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions        # Tutorial run functions
#------------------------------------------------------------------------------
casename=$(basename $PWD)
if test -f "$PWD/$casename.png"; then
    echo "$PWD/$casename.png"
else
    touch case.foam
    pvpython renderResults.py --decomposed $casename
    convert  $casename.png -transparent white -trim -resize 90% $casename.png
    echo "$PWD/$casename.png"
    #curl -s --location --request POST "https://api.imgbb.com/1/upload?expiration=600&key=${IMGBB_API_KEY}"\
    #--form "image=@./$casename.png" | jq .data.url | tee url
fi
#------------------------------------------------------------------------------
