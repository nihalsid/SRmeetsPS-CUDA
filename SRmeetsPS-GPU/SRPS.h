#pragma once
#include <opencv2/photo.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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
	void preprocessing();
};

