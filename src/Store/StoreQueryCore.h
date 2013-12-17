/*
 * QueryCore.h
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#ifndef QUERYCORE_H_
#define QUERYCORE_H_

#include "StoreQuery.h"
#include "StoreElement.h"
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

		/*
		 * Description:
		 * Method executing query with optional selected dataLocations
		 * If no dataLocation provided query is executed to all data in store
		 * Returns:
		 * 	size of data produced by query
		 * Output:
		 * 	result of query is returned to queryResult parameter
		 */
		size_t ExecuteQuery(void** queryResult, storeQuery* query, boost::container::vector<ullintPair>* dataLocationInfo = nullptr);

	private:
		/* DATA MANAGEMENT METHODS */
		size_t mapData(void** data, boost::container::vector<ullintPair>* dataLocationInfo);
		storeElement* decompressData(void* data, size_t size);
		void filterData(storeElement* elements, storeQuery* query);

		/* AGGREGATION MATHODS */
		void add(storeQuery* query);
	};

} /* namespace store */
} /* namespace ddj */
#endif /* QUERYCORE_H_ */
