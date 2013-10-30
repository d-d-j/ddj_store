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

namespace ddj {
namespace store {

class QueryMonitor
{
	QueryCore* _core;
public:
	QueryMonitor(CudaController* cudaController);
	virtual ~QueryMonitor();
};

} /* namespace store */
} /* namespace ddj */
#endif /* QUERYMONITOR_H_ */
