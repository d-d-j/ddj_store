/*
 * QueryCore.h
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#ifndef QUERYCORE_H_
#define QUERYCORE_H_

#include "../Cuda/CudaController.h"
#include "../Cuda/CudaIncludes.h"

namespace ddj {
namespace store {

	class StoreQueryCore
	{
		CudaController* _cudaController;
	public:
		StoreQueryCore(CudaController* cudaController);
		virtual ~StoreQueryCore();

		size_t SelectAll(void** queryResult);
	};

} /* namespace store */
} /* namespace ddj */
#endif /* QUERYCORE_H_ */
