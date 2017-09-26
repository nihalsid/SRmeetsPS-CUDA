#include <iostream>
#include "Utilities.h"
#include "SRPS.h"
int main() {
	DataHandler datahandler;
	datahandler.loadDataFromMatFiles("test_sf2.mat");
	SRPS srps(datahandler);
	srps.preprocessing();
	return 0;
}