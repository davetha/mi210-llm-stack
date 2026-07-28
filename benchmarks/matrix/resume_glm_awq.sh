#!/bin/sh
# Resume the GLM-4.6 AWQ download. It stopped at 167 of 189 GB with
# "errorCode=2 Timeout" on the last few shards -- transient network, not
# corruption. fetch_model.py keeps partial files, and aria2 --continue resumes
# them; anything incomplete WITHOUT an aria2 control file is purged rather than
# resumed, because a multi-connection partial can have holes in the middle and
# resuming by size would produce a full-sized file with zeroed gaps.
#
# Concurrency dropped from 8 to 3: the timeouts happened while three benchmark
# arms and another large download were competing for the same NVMe and link.
B=/mnt/llm-storage/bench-matrix
while ps -eo cmd | grep -q "[f]etch_model"; do sleep 60; done
echo "### resuming glm-awq $(date -u)"
cd $B/bin && python3 fetch_model.py bullpoint/GLM-4.6-AWQ $B/glm-awq --concurrent 3 >> $B/dl-big.log 2>&1 \
  && echo "OK glm-awq" >> $B/dl-big.status || echo "FAIL glm-awq (again)" >> $B/dl-big.status
echo "### glm-awq resume finished $(date -u)"; du -sh $B/glm-awq
