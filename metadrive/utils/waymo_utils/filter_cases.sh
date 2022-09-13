#!/usr/bin/env bash
for i in 0 1 2 3 4 5 6 7 8 9; do
    nohup python /home/lfeng/metadrive/metadrive/utils/waymo_utils/filter_cases.py /home/lfeng/waymo/scenarios_processed_${i} ${i} > ${i}.log 2>&1 &
done
