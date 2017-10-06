#include <iostream>
#include "Utilities.h"
#include "SRPS.h"
int main() {
	DataHandler dh;
	dh.loadDataFromMatFiles("mitten_12_16_sf2.mat");
	SRPS srps(dh);
	srps.execute();
	return 0;
}