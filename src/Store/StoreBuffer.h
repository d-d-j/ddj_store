/*
 *  StoreBuffer.h
 *
 *  Created by Karol Dzitkowski on 27.07.2013.
 *  Copyright (c) 2013 Karol Dzitkowski. All rights reserved.
 *
 *
 *      NAZEWNICTWO
 * 1. nazwy klas:  CamelStyle z dużej litery np. StoreController
 * 2. nazwy struktur camelStyle z małej litery np. storeElement
 * 3. nazwy pól prywatnych camelStyle z małej litery z podkreśleniem _backBuffer
 * 4. nazwy pól publicznych i zmiennych globalnych słowa rozdzielamy _ i z małych liter np. memory_available
 * 5. define z dużych liter i rozdzielamy _ np. BUFFER_SIZE
 * 6. nazwy metod publicznych z dużej litery CamelStyle np. InsertValue() oraz parametry funkcji z małych liter camelStyle np. InsertValue(int valToInsert);
 * 7. nazwy metod prywatnych z małej litery camelStyle
 * 8. nazwy funkcji "prywatnych" w plikach cpp z małej litery z _ czyli, insert_value(int val_to_insert);
 * 9. nazwy funkcji globalnych czyli w plikach .h najczęściej inline h_InsertValue() dla funkcji na CPU g_InsertValue() dla funkcji na GPU
 */

#ifndef DDJ_Store__DDJ_StoreBuffer_h
#define DDJ_Store__DDJ_StoreBuffer_h

#include "StoreElement.cuh"
#include "StoreTrunkInfo.h"
#include "StoreUploadCore.h"
#include "../BTree/BTreeMonitor.h"
#include "../Core/Logger.h"
#include "../Cuda/CudaController.h"
#include "../Cuda/CudaIncludes.h"
#include <boost/thread.hpp>

namespace ddj {
namespace store {

	/**
	* @class StoreBuffer
	* Coordinates inserts of new data to store and support info about data location.
	* Implements public method for inserting new data to store.
	* Implements public method for flushing data in buffer to a main memory of DB on GPU.
	* Implements public method for getting locations of data from provided time periods.
	* @see Insert()
	* @see Select()
	* @see Flush()
	*/
    class StoreBuffer : public boost::noncopyable
    {
		/* FIELDS */
		private:
			metric_type _metric;					/**< This buffer coordinates data only with this tag */
			btree::BTreeMonitor* _bufferInfoTreeMonitor;	/**< protects access to B+Tree structure */
			StoreUploadCore* _uploadCore;			/**< protects access to GpuUploadCore class */

			/* LOGGER & CONFIG */
			Logger _logger;
			Config* _config;

			/* BUFFERS */
			int _bufferElementsCount;		/**< how many elements are now in _buffer */
			int _backBufferElementsCount;	/**< how many elements are to upload in _backBuffer */
			int _bufferCapacity;
			size_t _bufferSize;
			storeElement* _buffer;		/**< main buffer where data is inserted */
			storeElement* _backBuffer;	/**< buffer to upload as trunk */
			boost::mutex _bufferMutex;
			boost::mutex _backBufferMutex;

		/* METHODS */
		public:
			/** StoreBuffer constructor.
			* It creates a new instance of BTreeMonitor class.
			*/
			StoreBuffer(metric_type metric, int bufferCapacity, StoreUploadCore* uploadCore);

			/** StoreBuffer destructor.
			* Releases BTreeMonitor.
			*/
			virtual ~StoreBuffer();

			/**
			 * @fn void Insert(storeElement* element);
			 * Function inserts storeElement to Buffer (stores data in DB on GPU).
			 * If buffer if full, buffer is uploaded to GPU. After uploading data to GPU and execution
			 * of Kernel which appends data to store array in GPU, infoElement is inserted to B+Tree structure
			 * in order to store location of a trunk in GPU.
			 * @param storeElement* a pointer to storeElement to store in our DB
			 */
			void Insert(storeElement* element);

			/**
			 * @fn void Flush();
			 * Forces StoreBuffer to switchBuffers and upload buffer to GPU.
			 */
			void Flush();

			/**
			 * Gets locations of trunks which contain data with time from any of specified time periods.
			 * Each StoreBuffer object is used to Insert and Select locations of data with one metric specified
			 * in StoreBuffer constructor. This method is using BTreeMonitor to select from B+Tree structure
			 * locations of data with his metric and from time periods.
			 * @param timePeriods [in] a boost vector of ullintPari structs indicating from which time periods
			 * data needs to be selected.
			 * @return a pointer to boost vector of ullintPair structs which stores locations of selected data in
			 * main memory of database on GPU.
			 */
			boost::container::vector<ullintPair>* Select(boost::container::vector<ullintPair> timePeriods);

		private:
			storeTrunkInfo* uploadBuffer();
			void insertToBtree(storeTrunkInfo* element);
			void switchBuffers();
			void uploadBufferToGPU();
    };

} /* end namespace store */
} /* end namespace ddj */

#endif /* defined(DDJ_Store__DDJ_StoreBuffer_h) */
