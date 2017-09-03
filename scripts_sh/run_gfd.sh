#!/bin/bash

TS=`date +"_%d%b%y_%H%M"`
fn=$(basename "$1")
fb=${fn%.*}
lg="/home/saguinag/Logs/"$fb$TS".log"
echo " "

bin/linux/pgd  -f $1 --macro ./Results/$fn".macro"

