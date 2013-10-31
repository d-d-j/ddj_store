/*
 * QueryCore.h
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#ifndef QUERYCORE_H_
#define QUERYCORE_H_

#include "../CUDA/CudaController.h"
#include "../Store/StoreIncludes.h"
#include "../CUDA/cudaIncludes.h"

namespace ddj {
namespace store {

	class QueryCore
	{
		CudaController* _cudaController;
	public:
		QueryCore(CudaController* cudaController);
		virtual ~QueryCore();

		void* GetAllData(size_t &size);
	};

} /* namespace store */
} /* namespace ddj */
#endif /* QUERYCORE_H_ */
