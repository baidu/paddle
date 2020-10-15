#!/bin/sh

lib=$1
if [ $# -ne 1 ]; then echo "No input library"; exit -1 ; fi

num_paddle_syms=$(nm -D ${lib} | grep paddle | wc -l)
num_google_syms=$(nm -D ${lib} | grep google | grep -v paddle | grep "T " | wc -l)

if [ $num_paddle_syms -le 0 ]; then echo "Have no paddle symbols"; exit -1 ; fi
if [ $num_google_syms -ge 1 ]; then echo "Have some google symbols"; exit -1 ; fi

exit 0
