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
}

QueryMonitor::~QueryMonitor()
{
	delete this->_core;
}

} /* namespace store */
} /* namespace ddj */
