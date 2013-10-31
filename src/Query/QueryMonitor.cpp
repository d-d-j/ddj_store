/*
 * QueryMonitor.cpp
 *
 *  Created on: 17-09-2013
 *      Author: ghashd
 */

#include "QueryMonitor.h"

namespace ddj {
namespace store {

QueryMonitor::QueryMonitor(CudaController* cudaController)
{
	Config* config = Config::GetInstance();
	this->_core = new QueryCore(cudaController);
	this->_sem = new Semaphore(config->GetValue("SIMUL_QUERY_COUNT"));
}

QueryMonitor::~QueryMonitor()
{
	delete this->_core;
}

storeElement* QueryMonitor::GetEverything(size_t& size)
{
	this->_sem->Wait();
	storeElement* result = (storeElement*)this->_core->GetAllData(size);
	this->_sem->Release();
	return result;
}

} /* namespace store */
} /* namespace ddj */
