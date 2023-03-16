# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def allreduce_allpairs(gpus, instances, protocol):
    # gpuIds = [1]                # GPU IDs that perform reduction, max hops =3
    gpuIds = [1, 2, 9, 10]      # GPU IDs that perform reduction, max hops =3
    # gpuIds = [1, 2]              # GPU IDs that perform reduction, max hops =3
    gpuIds = [0, 1, 2, 3, 8, 9, 10, 11]      # GPU IDs that perform reduction, max hops =4
    size = gpus       # number of GPUs in system, also, used as copy size in send 
    rsize = len(gpuIds) # Number of reducer ranks in the systems == number of chunks
    chunksperloop = size * rsize # Total number of chunks
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLProgram("allreduce_reduce_broadcast", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual, dependence_nop=True):
        
        # Each rank sends the nth chunk to the nth rank into its scratch space: chunk transpose operation
        for r1 in range(size):  # Each GPU in system, sends chunks to some "pre-determined" set of reducer GPUs 
            for r2 in range(rsize): # To all the reducer ranks (could be 1 or more - equal to #GPU)
                if r1 != gpuIds[r2]:
                    index = r2 * size # Index of the send chunk on source rank, destRankIndex X CopySize(#Chunks in 1 send cmd)
                    c = chunk(r1, Buffer.input, index, size=size) # Reference to the Source chunk
                    c.copy(gpuIds[r2], 'scratch', sendtb=gpuIds[r2], recvtb=r1)

        # Each reducer rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for r in range(rsize): # Go through each reducer rank in the syste, and perform reduction on data (respective data + scratch memory)
            for index in range(0, size * (size-1)): # Go through scratch memory, 0 to CopySize x #recieved chunks (== #GPU)
                    c = chunk(gpuIds[r], Buffer.input, r*size + (index % size))  # Destination fragments (0..15) on given rank 
                    c.reduce(chunk(gpuIds[r], 'scratch', index), sendtb=(index % size))
                                    # other fragment in scratch memory - where chunks from other ranks are recieved
        
        # Each reducer rank sends the fully reduced nth chunk to all other gpus 
        # Broadcast reduced chunk to other GPUs in the system
        for r1 in range(rsize): # Go through all reducer ranks (== 1 <= #GPUs), source rank
            for r2 in range(size):      # All destinations GPUs in the system
                if gpuIds[r1] != r2:
                    index = r1 * size  # ReducerRank X CopySize gives the index for source copy chunk
                    c = chunk(gpuIds[r1], Buffer.input, index, size) # Source chunk
                    c.copy(r2, Buffer.input, index, sendtb=r2, recvtb=gpuIds[r1]) # Destination chunk
                
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

allreduce_allpairs(args.num_gpus, args.instances, args.protocol)