# SRmeetsPS

This repository contains CUDA implementation  for the paper:  
Songyou Peng, Bjoern Haefner, Yvain Queau and Daniel Cremers, "**[Depth Super-Resolution Meets Uncalibrated Photometric Stereo](https://arxiv.org/abs/1708.00411)**", In IEEE Conference on Computer Vision (ICCV) Workshop, 2017.

Original implementation in MATLAB can be found [here](https://github.com/pengsongyou/SRmeetsPS).

## Citation
If you use this code, please cite the paper:
```sh
@inproceedings{peng2017iccvw,
 author =  {Songyou Peng and Bjoern Haefner and Yvain Qu{\'e}au and Daniel Cremers},
 title = {{Depth Super-Resolution Meets Uncalibrated Photometric Stereo}},
 year = {2017},
 booktitle = {IEEE International Conference on Computer Vision (ICCV) Workshop},
}
```
Contact **Songyou Peng** [:envelope:](mailto:psy920710@gmail.com) for questions, comments and reporting bugs.

## Building and Running

### Linux

Move to the linux build folder and compile using the makefile. Specifically, from the project root, execute the following commands:
``` 
cd build/linux
make
export LD_LIBRARY_PATH=../../opencv/lib:../../matio/lib
```

Following command line options are available while running:

| Option        | Description           | Default Value  |
| ------------- |-------------| -----|
|``--blockx``<br>``-x``      | CUDA kernel block's x dimension | 256 |
|``--blockx``<br>``-x``      | CUDA kernel block's y dimension      | 4 |
|``-d``<br>``--dsloc`` | Path to dataset as mat file or folder containing<br>images (depth images must be 16bit)      |    ``../../dataset/Images/Mitten`` |
|``-device``<br>``--g`` | CUDA device to run the application on | 0 |
|``-dstype``<br>``--t`` | Dataset type, can be as ``matlab`` for MAT file<br>input or ``images`` for images as input,with depth<br>images having bitdepth 16 | ``images`` |

#### Example commands 
When using MATLAB mat files as the input dataset, from the ``PROJECT_ROOT/build/linux`` folder, run the command
```
./bin/SRmeetsPS-CUDA --dstype="matlab" --dsloc="../../dataset/Matlab/mitten_sf2.mat"
```

When using image files as input, run
```
./bin/SRmeetsPS-CUDA --dstype="images" --dsloc="../../dataset/Images/Mitten"
```

## Benchmark comparison against Matlab implementation

![alt text](https://user-images.githubusercontent.com/932110/32146523-320647c4-bcd9-11e7-9098-e6ca43c38318.png "Small GPU")

![alt text](https://user-images.githubusercontent.com/932110/32146522-31e5fc1c-bcd9-11e7-8323-e39bc45454e2.png "Small GPU")
