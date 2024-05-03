# Scene Flow Zoo

Scene Flow Zoo is an open source scene flow benchmark and model zoo. Based on the [ZeroFlow codebase](http://github.com/kylevedder/zeroflow), it uses the [BucketedSceneFlowEval](https://github.com/kylevedder/BucketedSceneFlowEval) to load datasets and perform evaluation.

Currently, the Zoo supports the following datasets:

 - [Argoverse 2](https://www.argoverse.org/)
 - [Waymo Open](https://waymo.com/open/)

The Zoo supports the following methods:

 - Feed-forward
   - [FastFlow3D](https://arxiv.org/abs/2103.01306) / [FastFlow3D XL](https://vedder.io/zeroflow)
   - [ZeroFlow and ZeroFlow XL](https://vedder.io/zeroflow)
   - [DeFlow](https://arxiv.org/abs/2401.16122)
 - Test time optimization
   - [Neural Scene Flow Prior (NSFP)](https://arxiv.org/abs/2111.01253)
   - [Fast NSF](https://arxiv.org/abs/2304.09121)
   - [Liu et al. 2024](https://arxiv.org/abs/2403.16116)
 

If you use this codebase, please cite the following paper:

```
@article{vedder2024zeroflow,
    author    = {Kyle Vedder and Neehar Peri and Nathaniel Chodosh and Ishan Khatri and Eric Eaton and Dinesh Jayaraman and Yang Liu Deva Ramanan and James Hays},
    title     = {{ZeroFlow: Fast Zero Label Scene Flow via Distillation}},
    journal   = {International Conference on Learning Representations (ICLR)},
    year      = {2024},
}
```

If you use the evaluation results, please cite the following paper:

```
@misc{khatri2024trackflow,
    author = {Ishan Khatri and Kyle Vedder and Neehar Peri and Deva Ramanan and James Hays},
    title = {I Can't Believe It's Not Scene Flow!},
    journal = {arXiv},
    eprint = {2403.04739},
    year = {2024},
    pdf = {https://arxiv.org/abs/2403.04739}
}
```

If you use any of the methods in the Zoo, please cite the appropriate paper for that method.

## Pre-requisites / Getting Started

Read the [Getting Started](./GETTING_STARTED.md) doc for detailed instructions to setup the datasets and use the prepared docker environments.

## Pretrained weights

Trained weights are available for the following methods:
 
 - [ZeroFlow and FastFlow3D](https://github.com/kylevedder/zeroflow_weights)
 - [DeFlow](https://github.com/KTH-RPL/DeFlow)

## Visualizing results

The `visualization/visualize_flow.py` script can visualize the ground truth flow and the predicted flow for various methods. Note that the visualizer requires the ability to start an X window; the `./launch.sh` script on a headed machine will do this for you.

## Training a model

 Inside the main container (`./launch.sh`), run the `train_pl.py` with a path to a config (inside `configs/`) and optionally specify any number of GPUs (defaults to all GPUs on the system).

```
python train_pl.py <my config path> --gpus <num gpus>
```

The script will start by verifying the val dataloader works, and then launch the train job.

Note that config files specify the batch size _per GPU_, so the effective batch size will be `batch_size * num_gpus`. In order to replicate our results, you _must_ use the effective batch size of 64 for the normal sized FastFlow3D-style model and an effective batch size of 12 for the XL model. Our configs are setup to run on 4 x A6000s for the normal model and 6 x A6000s for the XL model. If your system differs, set the `accumulate_grad_batches` parameter in the config to accumulate gradients over multiple batches to reach the same size effective batch.

## Testing a model

Inside the main  (`./launch.sh`), run the `test_pl.py` with a path to a config (inside `configs/`), a path to a checkpoint, and the number of GPUs (defaults to a single GPU).

```
python test_pl.py <my config path> <my checkpoint path> --gpus <num gpus>
```


## Submitting to the [AV2 2024 Scene Flow competition](https://www.argoverse.org/sceneflow)

1. Dump the outputs of the model for the `test` split
    - Run `test_pl.py` with a dumper config that has the `save_output_folder` set to the desired output folder, and the `test` set as the `val_split` (e.g. `configs/fastflow3d/argo/bucketed_nsfp_distillation_3x_test_dumper`)
    - The dumper must be run with a single GPU (the default), as some batch entries may be skipped with multi-GPU inference.
2. Build the competition submission the output with `python av2_scene_flow_competition_submit.py <path/to/dumped/output/folder/>`
    - This will create a zip file as a sibling in the filesystem to the output folder, named after it.
3. Submit the zip to the competition website.
    - EvalAI's CLI must be used, as the zip file exceeds the limit for web uploads.
