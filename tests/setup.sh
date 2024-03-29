#!/bin/bash

# Setup fake environment


# Prepare /tmp/argoverse2_tiny/val
rm -rf /tmp/argoverse2_tiny /tmp/argoverse2_tiny.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14576619/argoverse2_tiny.zip -O /tmp/argoverse2_tiny.zip
unzip -q /tmp/argoverse2_tiny.zip -d /tmp/

# Prepare /tmp/argoverse2_small/val
rm -rf /tmp/argoverse2_small
echo "Downloading the 23 chunks of argoverse2_small"
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668591/argove01.zip -O /tmp/argoverse_small_part_01.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668605/argove02.zip -O /tmp/argoverse_small_part_02.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668609/argove03.zip -O /tmp/argoverse_small_part_03.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668613/argove04.zip -O /tmp/argoverse_small_part_04.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668619/argove05.zip -O /tmp/argoverse_small_part_05.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668624/argove06.zip -O /tmp/argoverse_small_part_06.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14672605/argove07.zip -O /tmp/argoverse_small_part_07.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668635/argove08.zip -O /tmp/argoverse_small_part_08.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668642/argove09.zip -O /tmp/argoverse_small_part_09.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668643/argove10.zip -O /tmp/argoverse_small_part_10.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668650/argove11.zip -O /tmp/argoverse_small_part_11.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14672612/argove12.zip -O /tmp/argoverse_small_part_12.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668654/argove13.zip -O /tmp/argoverse_small_part_13.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668656/argove14.zip -O /tmp/argoverse_small_part_14.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668657/argove15.zip -O /tmp/argoverse_small_part_15.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668658/argove16.zip -O /tmp/argoverse_small_part_16.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668660/argove17.zip -O /tmp/argoverse_small_part_17.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668661/argove18.zip -O /tmp/argoverse_small_part_18.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668663/argove19.zip -O /tmp/argoverse_small_part_19.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668665/argove20.zip -O /tmp/argoverse_small_part_20.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668667/argove21.zip -O /tmp/argoverse_small_part_21.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668668/argove22.zip -O /tmp/argoverse_small_part_22.zip
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14668671/argove23.zip -O /tmp/argoverse_small_part_23.zip
for i in {1..23}
do
  echo "Unzipping argoverse_small part $i"
  unzip -q /tmp/argoverse_small_part_$(printf %02d $i).zip -d /tmp/
done