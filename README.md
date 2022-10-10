# HMMM-Compiler
A C to Hmmm (https://github.com/bwiedermann/HMMM.js) Complier

## Setup Instructions
1. Clone the repo with `git clone --recurse-submodules https://github.com/kavidey/HMMM-Compiler.git` (you need to clone with submodules)
2. Make sure you have python 3.9 installed (I tested it with 3.9.11)
3. Install dependencies using your package manager of choice (ex: `pip install -r requirements.txt`)

## Usage
1. Compile a C file using `python compile.py <file.c>` (ex: `python compile.py test.c`)
   - this will print out the resulting Hmmm code
2. Copy that code into the online Hmmm interpreter (https://bwiedermann.github.io/HMMM.js/) and run it

## Compiler Documentation & Notes
https://docs.google.com/document/d/1o2OwMKZIm1G1sFX6xQV-RzaNGjPsZn7bYC6QkByNcyo/edit