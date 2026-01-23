#!/bin/bash
if [ $# -lt 1  ]
then
   echo 'usage: makepairs.sh <words-file>'
   exit -1
fi
awk '{ a[NR] = $0 } END { for (i=1;i<=NR;i++) for (j=i+1;j<=NR;j++) print a[i] "," a[j] }' $1
