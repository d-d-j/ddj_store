/*
 * GpuUploaderCore.h
 *
 *  Created on: Sep 1, 2013
 *      Author: parallels
 */

#ifndef GPUUPLOADERCORE_H_
#define GPUUPLOADERCORE_H_

#include "../Store/StoreIncludes.h"
#include "../Store/storeElement.h"

namespace ddj {
namespace store {

	typedef struct int2
	{
		int x;
		int y;
	} int2;

	/*! \fn  int2 UploadStoreElementsArrayToGpu(storeElement* elements, int elemCount)
	 * \brief Function which uploads an array of store elements to GPU and adds it to GPU array.
	 * \param elements an array of storeElements to upload.
	 * \param elemCount an integer (how many elements are in elements table).
	 * \return function returns a struct of two ints which contains first and last index in GPU table where data was stored.
	 */
	int2 UploadStoreElementsArrayToGpu(storeElement* elements, int elemCount);

} /* namespace store */
} /* namespace ddj */

#endif /* GPUUPLOADERCORE_H_ */
