# SRmeetsPS

This repository contains CUDA implementation  for the paper:  
Songyou Peng, Bjoern Haefner, Yvain Queau and Daniel Cremers, "**[Depth Super-Resolution Meets Uncalibrated Photometric Stereo](https://arxiv.org/abs/1708.00411)**", In IEEE Conference on Computer Vision (ICCV) Workshop, 2017.

Original implementation: https://github.com/pengsongyou/SRmeetsPS

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





### Building and running
In the root folder run
```
export LD_LIBRARY_PATH=./opencv/lib:./matio/lib
make
```

To run, the follwing options are available:
	--blockx, -x (value:256)
		block dimension x
	--blocky, -y (value:4)
		block dimension y
	-d, --dsloc (value:./dataset/Images/Mitten)
		path to dataset mat file or folder containing images
	--device, -g (value:0)
		cuda device to run the application on
	--dstype, -t (value:images)
		dataset type, can be matlab or images
	-h, --help, --usage
		print help

Example 1 from mat files :
```
build/SRmeetsPS-CUDA --dstype="matlab" --dsloc="./dataset/Matlab/mitten_sf2.mat"
```

Example 2 from png image files :
```
build/SRmeetsPS-CUDA --dstype="images" --dsloc="./dataset/Images/Mitten"
```
