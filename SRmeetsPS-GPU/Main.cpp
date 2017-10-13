#include <iostream>
#include "Utilities.h"
#include "SRPS.h"

int Preferences::blockX = 256;
int Preferences::blockY = 4;
int Preferences::deviceId = 0;

int main(int argc, char* argv[]) {	
	const cv::String keys =
		"{help h usage|| print help}"
		"{dstype t|matlab| dataset type, can be matlab or images }"
		"{dsloc d|| path to dataset mat file or folder containing images }"
		"{device g|0| cuda device to run the application on }"
		"{blockx x|256| block dimension x}"
		"{blocky y|4| block dimension y}"
		;
	cv::CommandLineParser parser(argc, argv, keys);
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}
	if (!parser.has("dsloc")) {
		parser.printMessage();
		return 0;
	}
	Preferences::blockX = parser.get<int>("blockx");
	Preferences::blockY = parser.get<int>("blocky");
	Preferences::deviceId = parser.get<int>("device");

	if (parser.get<cv::String>("dstype").compare("matlab") == 0) {
		MatFileDataHandler dh;
		dh.loadDataFromMatFiles(parser.get<cv::String>("dsloc").c_str());
		SRPS srps(dh);
		srps.execute();
	}
	else if (parser.get<cv::String>("dstype").compare("images") == 0) {
		ImageDataHandler dh;
		dh.loadDataFromImages(parser.get<cv::String>("dsloc").c_str());
		SRPS srps(dh);
		srps.execute();
	}
	return 0;
}
