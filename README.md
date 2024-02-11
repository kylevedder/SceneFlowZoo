# ZeroFlow: Fast Zero Label Scene Flow via Distillation

[Kyle Vedder](http://vedder.io), [Neehar Peri](http://www.neeharperi.com/), [Nathaniel Chodosh](https://scholar.google.com/citations?user=b4qKr7gAAAAJ&hl=en), [Ishan Khatri](https://ishan.khatri.io/), [Eric Eaton](https://www.seas.upenn.edu/~eeaton/), [Dinesh Jayaraman](https://www.seas.upenn.edu/~dineshj/), [Yang Liu](https://youngleox.github.io/), [Deva Ramanan](https://www.cs.cmu.edu/~deva/), and [James Hays](https://faculty.cc.gatech.edu/~hays/)

Project webpage: [vedder.io/zeroflow](http://vedder.io/zeroflow)

arXiv link: [arxiv.org/abs/2305.10424](http://arxiv.org/abs/2305.10424)

**News:**
- Jan 16th, 2024: ZeroFlow has been accepted to ICLR 2024!
- July 31st, 2023: The ZeroFlow XL student model is now **state-of-the-art** on the [Scene Flow Challenge](https://eval.ai/web/challenges/challenge-page/2010/overview)! See the [Getting Started](./GETTING_STARTED.md) document for details on setting up training on additional data.
 - June 18th, 2023: ZeroFlow was selected as a highlighted method in the CVPR 2023 _Workshop on Autonomous Driving_ [Scene Flow Challenge](https://eval.ai/web/challenges/challenge-page/2010/overview)!
 

**Citation:**

```
@article{Vedder2024zeroflow,
    author    = {Kyle Vedder and Neehar Peri and Nathaniel Chodosh and Ishan Khatri and Eric Eaton and Dinesh Jayaraman and Yang Liu Deva Ramanan and James Hays},
    title     = {{ZeroFlow: Fast Zero Label Scene Flow via Distillation}},
    journal   = {International Conference on Learning Representations (ICLR)},
    year      = {2024},
}
```

## Pre-requisites / Getting Started

Read the [Getting Started](./GETTING_STARTED.md) doc for detailed instructions to setup the AV2 and Waymo Open datasets and use the prepared docker environments.

## Pretrained weights

Trained weights from the paper are available for download from [this repo](https://github.com/kylevedder/zeroflow_weights).

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

Inside the main  (`./launch.sh`), run the `train_pl.py` with a path to a config (inside `configs/`), a path to a checkpoint, and the number of GPUs (defaults to a single GPU).

```
python test_pl.py <my config path> <my checkpoint path> --gpus <num gpus>
```


## Submitting to the [AV2 2024 Scene Flow competition](https://www.argoverse.org/sceneflow)

1. Dump the outputs of the model for the `test` split
    - Run a dumper config with the `save_output_folder` set to the desired output folder, and the `test` set as the `val_split` (e.g. `configs/fastflow3d/argo/bucketed_nsfp_distillation_3x_test_dumper`)
2. Build the competition submission the output with `python av2_scene_flow_competition_submit.py <path/to/dumped/output/folder/>`
3. Submit the `submission.zip` to the competition website.