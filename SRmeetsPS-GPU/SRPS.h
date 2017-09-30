#pragma once
#include "Utilities.h"
#include "cusparse_v2.h"
#include "thrust/replace.h"
#include "thrust/execution_policy.h"
#include "thrust/device_vector.h"
#include "device_launch_parameters.h"

class SRPS
{
private:
	DataHandler* dh;
public:
	SRPS(DataHandler& dh);
	~SRPS();
	void preprocessing();
};

