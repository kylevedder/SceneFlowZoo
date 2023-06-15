# ZeroFlow: Fast Zero Label Scene Flow via Distillation

[Kyle Vedder](http://vedder.io), [Neehar Peri](http://www.neeharperi.com/), [Nathaniel Chodosh](https://scholar.google.com/citations?user=b4qKr7gAAAAJ&hl=en), [Ishan Khatri](https://ishan.khatri.io/), [Eric Eaton](https://www.seas.upenn.edu/~eeaton/), [Dinesh Jayaraman](https://www.seas.upenn.edu/~dineshj/), [Yang Liu](https://youngleox.github.io/), [Deva Ramanan](https://www.cs.cmu.edu/~deva/), and [James Hays](https://faculty.cc.gatech.edu/~hays/)

Project webpage: [vedder.io/zeroflow](http://vedder.io/zeroflow)

arXiv link: [arxiv.org/abs/2305.10424](http://arxiv.org/abs/2305.10424)

**Citation:**

```
@article{Vedder2023zeroflow,
    author    = {Kyle Vedder and Neehar Peri and Nathaniel Chodosh and Ishan Khatri and Eric Eaton and Dinesh Jayaraman and Yang Liu Deva Ramanan and James Hays},
    title     = {{ZeroFlow: Fast Zero Label Scene Flow via Distillation}},
    journal   = {arXiv},
    year      = {2023},
}
```

## Pre-requisites / Getting Started

Read the [Getting Started](./GETTING_STARTED.md) doc for detailed instructions to setup the AV2 and Waymo Open datasets and use the prepared docker environments.

## Training a model

 Inside the main container (`./launch.sh`), run the `train_pl.py` with a path to a config (inside `configs/`) and optionally specify any number of GPUs (defaults to all GPUs on the system).

```
python train_pl.py <my config path> --gpus <num gpus>
```

The script will start by verifying the val dataloader works, and then launch the train job.

## Testing a model

Inside the main  (`./launch.sh`), run the `train_pl.py` with a path to a config (inside `configs/`), a path to a checkpoint, and the number of GPUs (defaults to a single GPU).

```
python test_pl.py <my config path> <my checkpoint path> --gpus <num gpus>
```

# Submitting to the AV2 Scene Flow competition

1. Dump the outputs of the model
    a. `configs/fastflow3d/argo/nsfp_distilatation_dump_output.py` to dump the `val` set result
    b. `configs/fastflow3d/argo/nsfp_distilatation_dump_output_test.py` to dump the `test` set result
2. Convert to the competition submission format (`competition_submit.py`)
3. Use official zip `make_submission_archive.py` file (`python /av2-api/src/av2/evaluation/scene_flow/make_submission_archive.py <path to step 2 results> /efs/argoverse2/test_official_masks.zip`)