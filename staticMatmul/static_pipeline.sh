#!/bin/bash

./gen-spirv.sh
./disas-spirv.sh

sed -i 's/Physical32/Physical64/g' matrixMultiplyStaticReadable.spv

./recompile_from_readable.sh

