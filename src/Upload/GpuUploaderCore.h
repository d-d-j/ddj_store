/*
 * GpuUploaderCore.h
 *
 *  Created on: Sep 1, 2013
 *      Author: parallels
 */

#ifndef GPUUPLOADERCORE_H_
#define GPUUPLOADERCORE_H_

#include "../StoreIncludes.h"

namespace ddj {
namespace store {

	/*!	\struct storeElement
	 	\brief this structs will be stored in GPU side array where all data will be stored.

	 	storeElement is the data structure send to this system by customers as a single
	 	time series event. It contains time as unsigned long long int, is tagged, has
	 	a value and belongs to certain series.
	 */
	typedef struct storeElement
	{
		public:
			/* FIELDS */
			int series;
			tag_type tag;
			ullint time;
			store_value_type value;

			/* CONSTRUCTORS */
			storeElement(){ series = 0; tag = 0; time = 0; value = 0; }
			storeElement(int _series, tag_type _tag, ullint _time, store_value_type _value)
			: series(_series), tag(_tag), time(_time), value(_value) {}
			storeElement(const storeElement& elem)
			{
				this->series = elem.series;
				this->tag = elem.tag;
				this->time = elem.time;
				this->value = elem.value;
			}
			~storeElement(){}
	} storeElement;

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
