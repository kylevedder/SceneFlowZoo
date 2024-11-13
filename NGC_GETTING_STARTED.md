# NGC Getting Started

## Preparing docker images

NGC launches jobs using docker images that live inside Nvidia's corporate systems. Thus, in order to run our jobs using the standard SceneFlowZoo docker images, we must rehost them on nvcr.io. To do this, pull the public image locally (or build it locally), retag it for whatever the image will be called internally, and than uploading it to nvcr.io like so:

```
docker image pull kylevedder/zeroflow_bucketed
docker image tag kylevedder/zeroflow_bucketed:latest nvcr.io/nvidian/scene-flow-zoo:latest
docker image push nvcr.io/nvidian/scene-flow-zoo:latest
```

## Preparing a "login node"

NGC does not have a "login node" like SLURM does, so you must launch your own job that you then connect to if you want to do operations to modify your workspaces. To do this, run

```
ngc batch run --instance dgx1v.16g.1.norm --team swaiinf --name 'NGC-tensorboard-env-kvedder' --image "nvcr.io/nvidian/scene-flow-zoo:latest" --result /result --workspace zhiding_intern_kyle:/workspace --workspace zhiding_intern_kyle_argoverse2:/workspace_argoverse2 --commandline "/workspace/entrypoint.sh; sleep 167h" --port 6006 --priority HIGH
```

which launches a job on a single V100 16GB node (`dgx1v.16g.1.norm`). This job will run the entrypoint script upon launching, which configures the proper symlinks to make all the files appear where client code expects (`/workspace/entrypoint.sh`) and then sleeps for 167 hours, the maximum amount of time a job can be run on NGC, meaning about once a week you will need to redo this operation.

Running this `ngc batch run` command will provide job information that contains a job ID. This ID can then be used to connect to the node with

```
ngc batch exec <jobid>
```

to get an interactive shell. Note that, unlike the launch command, you _must_ be connected to the Nvidia VPN or this operation will hang.

## Preparing per sequence runs

A major aspect of test time optimization methods is for them to handle different sequences entirely separately. For datasets like Argoverse 2 or Waymo Open, these sequences have different lengths, meaning that each optimization job launched needs to be configured to load that sequence name and length.

### Precomputing sequence lengths

To support this, we precompute the sequence lengths into a `.json` file. As an example:

```
python data_prep_scripts/argo/make_sequence_lengths_metadata.py /efs/argoverse2/val
```

will produce `/efs/argoverse2/val_sequence_lengths.json`. **NB:** This should already be done for Argoverse2 and Waymo Open, it is merely included for completeness.

### Preparing per-sequence jobs

We then leverage this sequence length file to generate per-sequence custom configs, based on a base config. In this example, we are running GIGACHAD on the Argoverse 2 train split. To do this, we use the base GIGACHAD config file 

```
/workspace/code/scene_flow_zoo/configs/gigachad_nsf/argo/noncausal_minibatched/train.py
```

which specifies the model configuration (e.g. flow field network depth) and basic data loader configs. The sequence name and sequence lengths of the dataloader will be overwritten with the per-sequence custom config. **NB: the full raw path (starting with `/workspace`) to the config must be specified. This is to prevent resolution errors later.**

```
python data_prep_scripts/split_by_sequence.py /workspace/code/scene_flow_zoo/configs/gigachad_nsf/argo/noncausal_minibatched/train.py /efs/argoverse2/train_sequence_lengths.json launch_files/ ngc gigachad_train "--label ml__gigachad" "--team swaiinf" "--image 'nvcr.io/nvidian/scene-flow-zoo:latest'" "--workspace zhiding_intern_kyle /workspace" "--workspace zhiding_intern_kyle_argoverse2 /workspace_argoverse2"
```
Once the script finishes, the last line will be a path to `launch_files/*/launch_all.sh`. This `launch_all.sh` file contains all the `ngc batch run` commands to launch the jobs on your Nvidia provided machine. 

Before you can do this, you must setup your local machine for copying and 

## Copying to (and from) to your Nvidia provided machine

To move data between workspaces and your Nvidia machine, you have to intermediary through another storage medium (for some reason :eyeroll:). This is done via `rclone sync` calls to and from this intermediary.

### Preparing your Nvidia machine

**NB: This need only be done once.**

To prepare your local machine to sync to this, ensure that `rclone` is installed, and then inside your "login node" `exec`'d instance on NGC, run

```
cat ~/.config/rclone/rclone.conf
```

and paste those results into the same path on your local machine. This should now allow you to `rclone sync` to and from the named storage `pbss-zhiding-intern-kyle`.


### Copying `launch_all.sh` files to your Nvidia machine


#### Pushing to intermediary storage

On the "login node", run 

```
python /workspace/sync_launch_files.py | sort -r  > rclone_all.sh
bash rclone_all.sh
```

This will run the `rclone` sync commands for _all_ launch directories. The `sort -r` is intended to ensure your new one is hopefully one of the first sync'd, but you must make sure an actual copy occurs before you move on to the next step.

#### Pulling from intermediary storage

On your Nvidia machine, run

```
rclone lsf --dirs-only pbss-zhiding-intern-kyle: | grep "launch_all_" | sort | while read -r dir; do
    rclone sync -P pbss-zhiding-intern-kyle:"$dir" "$dir" -v
done
```

This will iterate through all the `launch_all_*` directories and sync them locally.

## Launching (and monitoring) all jobs

With the `launch_all.sh` copied to your machine, open a terminal multiplexer session (e.g. `screen`) to keep a long-running job (our monitor script). Inside that session, run

```
python util_scripts/ngc_monitor.py --ssh-host localhost path/to/launch_all.sh
```

This will run kickoff and monitor the status of all your jobs, restarting as needed. It will restart all failed jobs with 32GB instead of 16GB cards, but it is incumbent upon you to make sure it's not failing for other reasons.

To check on the status of your jobs, run

```
python util_scripts/ngc_job_status.py --ssh-host localhost
```

to automatically parse your job history from the last 7 days and show you their statuses. **NB: both of these tools check job state by using a 7 day history. If left running longer than 7 days, this will mean it believes jobs that have already completed have not been run and will relaunch them.**