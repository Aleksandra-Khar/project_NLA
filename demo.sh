#!/bin/bash
set -e

make
if [ ! -e text8 ]; then
	
  unzip text8.zip

fi

make



#bash gloveca.sh  text01 5  30  100
bash gloveca.sh  text01  5  30  10
bash gloveca.sh  text01  5  30  100

#bash tailcut.sh  text01 5  24  20
#bash tailcut.sh  text8  5  24  8000


