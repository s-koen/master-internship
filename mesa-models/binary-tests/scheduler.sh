#! /bin/bash

echo "moving to 3M-115R-q0.8"
cd binary-rees-3M-115R-q0.8/

echo "starting simulation"
./clean
./mk
./rn

echo "simulation terminated"

echo "moving to 3M-59R-q0.8"
cd ..
cd binary-rees-3M-59R-q0.8/

echo "starting simulation"
./clean
./mk
./rn

echo "simulation terminated"

echo "moving to 3M-115-q1"
cd ..
cd binary-rees-3M-115-q1/

echo "starting simulation"
./clean
./mk
./rn

echo "simulation terminated"
