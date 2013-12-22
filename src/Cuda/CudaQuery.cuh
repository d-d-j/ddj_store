#ifndef CUDAQUERY_CUH_
#define CUDAQUERY_CUH_

#include "../Store/StoreElement.h"
#include "../Store/StoreQuery.h"
#include "../Store/StoreTypedefs.h"

extern "C" {

size_t gpu_filterData(ddj::store::storeElement* elements, int elemCount, ddj::store::storeQuery* query);

}


#endif /* CUDAQUERY_CUH_ */
