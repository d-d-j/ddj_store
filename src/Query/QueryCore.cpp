/*
 * QueryCore.cpp
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#include "QueryCore.h"

namespace ddj {
namespace store {

QueryCore::QueryCore(CudaController* cudaController)
{
	this->_cudaController = cudaController;
}

QueryCore::~QueryCore(){}

} /* namespace store */
} /* namespace ddj */
