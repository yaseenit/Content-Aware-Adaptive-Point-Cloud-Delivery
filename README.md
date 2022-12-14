# Content-Aware Adaptive Point Cloud Delivery

## Abstract 

Point clouds are an important enabler for a wide range of applications in various domains, including autonomous vehicles and virtual reality applications. Hence, the practical applicability of point clouds is gaining increasing importance and presenting new challenges for communication systems where large amounts of data need to be shared with low latency. Point cloud content can be very large, especially when multiple objects are involved in the scene. Major challenges of point clouds delivery are related to streaming in bandwidth-constrained networks and to resource-constrained devices. In this work, we are exploiting object-related knowledge, i.e., content-driven metrics, to improve the adaptability and efficiency of point clouds transmission. This study proposes applying a 3D point cloud semantic segmentation deep neural network and using object-related knowledge to assess the importance of each object in the scene. Using this information, we can semantically adapt the bit rate and utilize the available bandwidth more efficiently. The experimental results conducted on a real-world dataset showed that we can significantly reduce the requirement for multiple object point cloud transmission with limited quality degradation compared to the baseline without modifications.

## Preview
|Before|After|
|-|-|
|![Before](before.gif)|![After](after.gif)|

## Quick Start Guide

Detailed descriptions for setup and usage can be found in the individual projects README files. Here is the quick start version:


### Training
- Setup the "SalsaNext" project per README
- Open a terminal and `cd` into `train/task/semantic`
- You may have to set `CUDA_VISIBLE_DEVICES` before the next command to be able to train on GPU(s) (e.g. `export CUDA_VISIBLE_DEVICES="1,2"`)
- Run the training with the following command:

```bash
python ./train.py -d /path/to/dataset -s train -ac  /abs/path/to/project/salsanext.yml -dc /abs/path/to/project/train/tasks/semantic/config/labels/semantic-kitti.yaml -l /abs/path/to/project/logs -u false -n iou_class-wce_class
```

After the training has finished the model can be found in the `logs` directory.

### Inference
- Setup the project "SalsaNext" per README
- Open a terminal and `cd` into `train/task/semantic`
- You may have to set `CUDA_VISIBLE_DEVICES` before the next command to be able to infere on GPU(s) (e.g. `export CUDA_VISIBLE_DEVICES="1,2"`)
- Run the inference with the following command:

```bash
python ./infer.py -d /path/to/dataset -s valid --log=/path/for/predictions -u false -m  --metrics_log /path/for/metrics.csv --metrics_group_size 10
```


### Pose Estimation
- Setup the "semantic-kitti-api" project per README
- Open a terminal and `cd` into the project directory
- Run:
```bash
python ./visualize.sh\
    --dataset "/path/to/dataset"\
    --config "./config/semantic-kitti-crit.yaml",
    --sequence "08"\
    --additional_sequences_paths "/path/to/crit1" "/path/to/crit2" "/path/to/crit3"\
    --additional_sequences_labels "301" "302" "303"\
    --additional_sequences_strides "1" "2" "5"\
    --additional_sequences_strides_compensation "one of drop/freeze/pose"\
    --whitelist "301" "302" "303"\
    --log_filtered "ply"
```

The visualizer will start. You can control it with the following keys:
|Key|Function|
|-|-|
|b|back (previous scan)|
|n|next (next scan)|
|l|toggle loop|
|f|toggle fast forward|
|q|quit (exit program)|

If a log directory is set it will log the results at each step.


### PSNR Calculation

We used the metrics tool provided by MPEG in the [mpeg-pcc-tmc2](https://github.com/MPEGGroup/mpeg-pcc-tmc2) repository.

- Clone and build the repository by the given instructions
- Use the compiled `PCCAppMetrics` program from the `/bin/Release` directory to compare point clouds.

### Disclaimer

We based our code heavily on [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext) and [API for SemanticKITTI](https://github.com/PRBonn/semantic-kitti-api). Give them a visit too!