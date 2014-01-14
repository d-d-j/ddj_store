/*
 * main.cpp
 *
 *  Created on: Aug 10, 2013
 *      Author: Karol Dzitkowski
 *
 *      NAZEWNICTWO
 * 1. nazwy klas:  CamelStyle z dużej litery np. StoreController
 * 2. nazwy struktur camelStyle z małej litery np. storeElement
 * 3. nazwy pól prywatnych camelStyle z małej litery z podkreśleniem _backBuffer
 * 4. nazwy pól publicznych i zmiennych globalnych słowa rozdzielamy _ i z małych liter np. memory_available
 * 5. define z dużych liter i rozdzielamy _ np. BUFFER_SIZE
 * 6. nazwy metod publicznych z dużej litery CamelStyle np. InsertValue() oraz parametry funkcji z małych liter camelStyle np. InsertValue(int valToInsert);
 * 7. nazwy metod prywatnych z małej litery camelStyle
 * 8. nazwy funkcji "prywatnych" w plikach cpp z małej litery z _ czyli, insert_value(int val_to_insert);
 * 9. nazwy funkcji globalnych czyli w plikach .h najczęściej inline h_InsertValue() dla funkcji na CPU g_InsertValue() dla funkcji na GPU
 */

#include "Node.h"
#include "Core/Logger.h"
#include "Core/Config.h"
#include <gtest/gtest.h>
#include <cmath>
#include "Cuda/CudaCommons.h"

void InitializeLogger() {
	log4cplus::initialize();
	LogLog::getLogLog()->setInternalDebugging(true);
	PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT("ddj_logger.prop"));
}

// TODO: Refactor!!
void loadExampleData(ddj::Node* node)
{
	int N = 2000000;
	ddj::store::storeElement* elem = nullptr;
	ddj::store::CudaCommons cudaC;
	int devId = cudaC.SetCudaDeviceWithMaxFreeMem();
	int i1 = 0;
	int i2 = 0;
	int i3 = 0;
	int i4 = 0;
	for(; i1<N; i1++)
	{
		elem = new ddj::store::storeElement(0, 0, i1, std::sin(i1/100.0f*M_PI));
		ddj::task::taskRequest req1(0*N+i1, ddj::task::Insert, devId, sizeof(storeElement), elem);
		node->CreateTask(req1);
		if(i1%99999==0) printf("Inserted %d from %d elements\n", 0*N+i1+1, 4*N);
	}
	devId = cudaC.SetCudaDeviceWithMaxFreeMem();
	for(; i2<N; i2++)
	{
		elem = new ddj::store::storeElement(1, 0, i2, std::cos(i2/100.0f*M_PI));
		ddj::task::taskRequest req2(1*N+i2, ddj::task::Insert, devId, sizeof(storeElement), elem);
		node->CreateTask(req2);
		if(i2%99999==0) printf("Inserted %d from %d elements\n", 1*N+i2+1, 4*N);
	}
	devId = cudaC.SetCudaDeviceWithMaxFreeMem();
	for(; i3<N; i3++)
	{
		elem = new ddj::store::storeElement(2, 1, i3, 1.0f);
		ddj::task::taskRequest req3(2*N+i3, ddj::task::Insert, devId, sizeof(storeElement), elem);
		node->CreateTask(req3);
		if(i3%99999==0) printf("Inserted %d from %d elements\n", 2*N+i3+1, 4*N);
	}
	devId = cudaC.SetCudaDeviceWithMaxFreeMem();
	for(; i4<N; i4++)
	{
		elem = new ddj::store::storeElement(3, 1, i4, i4*1.0f);
		ddj::task::taskRequest req4(3*N+i4, ddj::task::Insert, devId, sizeof(storeElement), elem);
		node->CreateTask(req4);
		if(i4%99999==0) printf("Inserted %d from %d elements\n", 3*N+i4+1, 4*N);
	}
	printf("All data inserted!\n");
}

// TODO: Refactor!!
int main(int argc, char* argv[])
{
	ddj::Config::GetInstance();
	InitializeLogger();
	Logger logger = Logger::getRoot();
	bool enableExampleData = false;

	if (argc >= 2)
	{
		if(!strcmp(argv[1], "--exampleData"))
		{
			enableExampleData = true;
		}
		else
		{
			Logger::getRoot().removeAllAppenders();
			::testing::InitGoogleTest(&argc, argv);
			if(!strcmp(argv[1], "--test"))
			{
				::testing::GTEST_FLAG(filter) = "*Test*";
			}
			else if(!strcmp(argv[1], "--performance"))
			{
				::testing::GTEST_FLAG(filter) = "*StorePerformance*";
				::testing::FLAGS_gtest_repeat = 1;
			}
			return RUN_ALL_TESTS();
		}
	}
	else
	{
		Logger::getInstance(LOG4CPLUS_TEXT("test")).removeAllAppenders();
	}

	LOG4CPLUS_INFO(logger, LOG4CPLUS_TEXT("Node main application started"));

	ddj::Node n;
	if(enableExampleData) loadExampleData(&n);
	getchar();
	return EXIT_SUCCESS;
}

