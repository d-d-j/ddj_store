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
#include "signal.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

po::variables_map initialize_options(int argc, char* argv[])
{
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "shows help message")
	    ("test", "runs all unit tests")
	    ("performance", "runs all performance tests")
	    ("exampleData", po::value<int>(), "inserts N elements to 4 different series")
	;

	po::variables_map options;
	po::store(po::parse_command_line(argc, argv, desc), options);
	po::notify(options);

	if (options.count("help")) {
	    std::cout << desc << std::endl;
	    exit(0);
	}

	return options;
}

void initialize_logger()
{
	log4cplus::initialize();
	LogLog::getLogLog()->setInternalDebugging(true);
	PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT("ddj_logger.prop"));
}

int wait_to_terminate()
{
	//wait for SIGINT
	int sig_number;
	sigset_t signal_set;
	sigemptyset (&signal_set);
	sigaddset (&signal_set, SIGINT);
	sigwait (&signal_set, &sig_number);

	return EXIT_SUCCESS;
}

int configure_tests(po::variables_map options, int argc, char* argv[])
{
	bool enableTest = false;
	if(options.count("test"))
	{
		::testing::GTEST_FLAG(filter) = "*Test*";
		enableTest = true;
	}
	else if(options.count("performance"))
	{
		::testing::GTEST_FLAG(filter) = "*Performance*";
		enableTest = true;
	}
	if(enableTest)
	{
		Logger::getRoot().removeAllAppenders();
		::testing::InitGoogleTest(&argc, argv);
		::testing::FLAGS_gtest_repeat = 1;
		return 1;
	}
	return 0;
}

float sinElem(int i){ return std::sin(i/100.0f*M_PI); }
float cosElem(int i){ return std::cos(i/100.0f*M_PI); }
float constElem(int i){ return 1.0f; }
float linElem(int i){ return i; }

/* 4 series of data will be inserted to store
 *
 *	tag		metric		data
 * 	0		0			sin
 *	1		0			cos
 *	2		1			1.0
 *	3		1			y = x
 *
 * N - number of elements to insert to each series
 */
void load_example_data(ddj::Node* node, int N, int tag, int metric, float (*f) (int))
{
	// Select best device
	ddj::store::CudaCommons cudaC;
	int devId = cudaC.SetCudaDeviceWithMaxFreeMem();

	for(int i=0; i<N; i++)
	{
		ddj::store::storeElement* elem = new ddj::store::storeElement(tag, metric, i, f(i));
		ddj::task::taskRequest req(i, ddj::task::Insert, devId, sizeof(storeElement), elem);
		node->CreateTask(req);
		if(i%99999==0) printf("Inserted %d from %d elements\n", i+1, N);
	}
}

void configure_system_after_start(po::variables_map options, ddj::Node* node)
{
	if(options.count("exampleData"))
	{
		int N = options["exampleData"].as<int>();
		printf("Inserting %d sin elements...", N);
		load_example_data(node, N, 0, 0, &sinElem);
		printf("Inserting %d cos elements...", N);
		load_example_data(node, N, 1, 0, &cosElem);
		printf("Inserting %d const elements...", N);
		load_example_data(node, N, 2, 1, &constElem);
		printf("Inserting %d linear elements...", N);
		load_example_data(node, N, 3, 1, &linElem);
		printf("INSERTED ALL EXAMPLE DATA!\n");
	}
}

int main(int argc, char* argv[])
{
	ddj::Config::GetInstance();
	initialize_logger();
	Logger logger = Logger::getRoot();
	auto options = initialize_options(argc, argv);

	if(configure_tests(options, argc, argv)) return RUN_ALL_TESTS();

	// HERE WHOLE NODE SYSTEM STARTS
	ddj::Node n;
	LOG4CPLUS_INFO(logger, LOG4CPLUS_TEXT("NODE START SUCCESS"));

	configure_system_after_start(options, &n);

	return wait_to_terminate();
}

