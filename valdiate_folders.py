from pathlib import Path

dataset_folders = Path("/efs/argoverse_lidar/train/").glob("*")

len_set = set()
for folder in dataset_folders:
    folder_len = len(list((folder / "sensors" / "lidar").glob("*")))
    if folder_len < 296:
        print(folder)
    len_set.add(folder_len)

print(len_set)
    