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
	this->_core = new QueryCore(cudaController);
	this->_sem = new Semaphore(_config->GetIntValue("SIMUL_QUERY_COUNT"));
}

QueryMonitor::~QueryMonitor()
{
	delete this->_core;
}

size_t QueryMonitor::SelectAll(storeElement** queryResult)
{
	this->_sem->Wait();
	size_t size = this->_core->SelectAll((void**)queryResult);
	this->_sem->Release();
	return size;
}

} /* namespace store */
} /* namespace ddj */
