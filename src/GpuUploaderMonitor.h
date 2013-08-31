/*
 * GpuUploaderMonitor.h
 *
 *  Created on: Aug 31, 2013
 *      Author: Karol Dzitkowski
 */

#include "StoreIncludes.h"

#ifndef GPUUPLOADERMONITOR_H_
#define GPUUPLOADERMONITOR_H_

namespace ddj {
namespace store {

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

	struct int2
	{
		int x;
		int y;
	};

	class GpuUploaderMonitor
	{
		private:
			boost::thread* _threadUploader;

			boost::condition_variable _condUploader;
			boost::condition_variable _condSynchronization;

			boost::mutex _mutexSynchronization;
			boost::mutex _mutexUploader;

		public:
			GpuUploaderMonitor();
			virtual ~GpuUploaderMonitor();

		private:
			/*! \fn  int2 GpuUploaderMonitor::uploadArrayToGPU(storeElement* elements, int elemCount)
			 * \brief Function which uploads an array of store elements to GPU and adds it to GPU array.
			 * \param elements an array of storeElements to upload.
			 * \param elemCount an integer (how many elements are in elements table).
			 * \return function returns a struct of two ints which contains first and last index in GPU table where data was stored.
			 */
			int2 uploadArrayToGPU(storeElement* elements, int elemCount);
	};

} /* namespace store */
} /* namespace ddj */
#endif /* GPUUPLOADERMONITOR_H_ */
