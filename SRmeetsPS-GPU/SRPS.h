#pragma once
#include <opencv2/photo/photo.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tuple>
#include <algorithm>
#include "Utilities.h"
#include "devicecalls.cuh"

class SRPS
{
private:
	DataHandler* dh;
public:
	SRPS(DataHandler& dh);
	~SRPS();
	void execute();
};

