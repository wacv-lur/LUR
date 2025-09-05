### NuScenes driver action and intention recognition labels

First download the NuScenes dataset:

```wget https://motional-nuscenes.s3.amazonaws.com/public/nuimages-v1.0/nuimages-v1.0-all-sweeps-cam-front.tgz```

#### Included classes:
Left turn, right turn, left lane change, right lane change, moving forward, waiting.

#### Dataloader example:
For an example torch dataloader implementation, please refer to the [dataloader.py](dataloader.py) file.
