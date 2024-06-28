#!/usr/bin/bash
#manim --disable_caching -qh -p bayesian.py
source .venv/bin/activate
manim -qh -p bayesian.py
manim-slides convert --to html -c progress=true -c controls=true -cslide_number=true "Bayes" "Bayes.html"
./node_modules/html-inject-meta/cli.js < Bayes.html  > index.html
