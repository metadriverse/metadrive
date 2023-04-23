#!/usr/bin/env bash
for i in 0 1 2 3 4 5 6 7 8 9; do
    nohup python /home/qyli/metadrive/metadrive/utils/waymo/script/convert_waymo_to_metadrive.py ./scenario_${i} > ${i}.log 2>&1 &
done
