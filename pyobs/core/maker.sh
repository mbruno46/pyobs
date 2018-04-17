#!/bin/bash

GCC=$1

uname="$(uname -s)"
case "${uname}" in
	Linux*)     MACHINE=Linux;;
       	Darwin*)    MACHINE=Mac;;
	*)          MACHINE="UNKNOWN:${uname}"
esac
echo ${MACHINE}

${GCC} -c -fPIC _libcore.c -o _libcore.o

if [ "${MACHINE}" = "Linux" ]; then
	${GCC} -shared -Wl,-soname,libcore.so -o _libcore.so -lm _libcore.o
elif [ "${MACHINE}" = "Mac" ]; then
	${GCC} -shared -Wl,-install_name,libcore.so -o _libcore.so -lm _libcore.o
fi

rm _libcore.o
