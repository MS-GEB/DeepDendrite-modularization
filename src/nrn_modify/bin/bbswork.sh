#!/bin/sh

prefix=/home/hegan/DeepDendrite
exec_prefix=/home/hegan/DeepDendrite/x86_64
NRNBIN=${exec_prefix}/bin
ARCH=x86_64
MODSUBDIR=x86_64
NEURONHOME=/home/hegan/DeepDendrite/share/nrn

cd $1

if [ -x ${MODSUBDIR}/special ] ; then
	program="./${MODSUBDIR}/special"
else
	program="${NRNBIN}/nrniv"
fi

hostname
pwd
shift
shift
echo "time $program $*"
time $program $*

