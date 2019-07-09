#!/bin/bash

export PRESTO=/opt/presto
export TEMPO=/opt/tempo
export PGPLOT_DIR=/usr/lib/pgplot5

if [ -z $PATH ]; then
    export PATH=$PRESTO/bin
else
    export PATH=$PATH:$PRESTO/bin
fi

if [ -z $LD_LIBRARY_PATH ]; then
    export LD_LIBRARY_PATH=$PRESTO/lib
else
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PRESTO/lib
fi

if [ -z $PYTHONPATH ]; then
    export PYTHONPATH=$PRESTO/lib/python
else
    export PYTHONPATH=$PYTHONPATH:$PRESTO/lib/python
fi
