#include <iostream>
#include "Utilities.h"
#include "SRPS.h"
int main(int argc, char* argv[]) {
	DataHandler dh;
	dh.loadDataFromMatFiles(argv[1]);
	SRPS srps(dh);
	srps.execute();
	return 0;
}
