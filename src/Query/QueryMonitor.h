/*
 * QueryMonitor.h
 *
 *  Created on: 17-09-2013
 *      Author: ghashd
 */

#ifndef QUERYMONITOR_H_
#define QUERYMONITOR_H_

#include "../CUDA/CudaController.h"
#include "QueryCore.h"
#include "../Store/storeElement.h"
#include "../Helpers/Semaphore.h"
#include "../Helpers/Config.h"

namespace ddj {
namespace store {

class QueryMonitor
{
	QueryCore* _core;
	Semaphore* _sem;
public:
	QueryMonitor(CudaController* cudaController);
	virtual ~QueryMonitor();
	size_t SelectAll(storeElement** queryResult);
};

} /* namespace store */
} /* namespace ddj */
#endif /* QUERYMONITOR_H_ */
