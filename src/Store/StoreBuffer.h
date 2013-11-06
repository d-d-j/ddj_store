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

#include "../BTree/BTreeMonitor.h"
#include "../GpuUpload/GpuUploadMonitor.h"
#include "storeElement.h"
#include "infoElement.h"
#include "../CUDA/cudaIncludes.h"

namespace ddj {
namespace store {

	/*! \class StoreBuffer
	 \brief Coordinates inserts of new data to store and support info about data location

	 Implement async function of inserting new data to store (UploaderThread)
	 Implement sync function of getting trunks with their location on GPU with this tag
	 and specified time (from - to)
	*/
    class StoreBuffer
    {
		/* FIELDS */
		private:
			tag_type _tag;	/**< This buffer coordinates data only with this tag */
			BTreeMonitor* _bufferInfoTreeMonitor;	/**< protects access to B+Tree structure */
			GpuUploadMonitor* _gpuUploadMonitor;	/**< protects access to GpuUploadCore class */

			/* UPLOADER THREAD */
			boost::thread* _uploaderThread;	/**< uploads _backBuffer to GPU and stores info in B+Tree */
			boost::mutex _uploaderMutex;
			boost::condition_variable _uploaderCond;
			boost::barrier* _uploaderBarrier;

			/* BUFFERS */
			int _bufferElementsCount;	/**< how many elements are now in _buffer */
			int _backBufferElementsCount;	/**< how many elements are to upload in _backBuffer */
			bool _areBuffersSwitched;	/**< true if _backBuffer is ready to upload and haven't been yet */
			boost::array<storeElement, STORE_BUFFER_SIZE> _buffer;	/**< main buffer where data is inserted */
			boost::array<storeElement, STORE_BUFFER_SIZE> _backBuffer;	/**< buffer to upload as trunk */

		/* METHODS */
		public:
			//! StoreBuffer constructor.
			/*!
			  It creates a new instance of BTreeMonitor class and starts Upload thread
			*/
			StoreBuffer(tag_type tag, GpuUploadMonitor* gpuUploadMonitor);

			//! BTreeMonitor destructor.
			/*!
		  	  releases BTreeMonitor and stops (+ joins) UploaderThread
			*/
			virtual ~StoreBuffer();

			/*! \fn  void Insert(storeElement* element);
			 * \brief Function inserts storeElement to Buffer (stores data in DB on GPU)
			 *
			 * If buffer if full or Flush method is executed, buffer is uploaded to GPU.
			 * After uploading data to GPU and execution of Kernel which appends data to
			 * store array in GPU, infoElement is inserted to B+Tree structure in order
			 * to store location of a trunk in GPU.
			 * \param storeElement* a pointer to storeElement to store in our DB
			 */
			void Insert(storeElement* element);

			/*! \fn  void Flush();
			 * \brief Forces StoreBuffer to switchBuffers and upload buffer to GPU
			 */
			void Flush();

		private:
			infoElement* uploadBuffer();
			void insertToBtree(infoElement* element);
			void switchBuffers();
			void uploaderThreadFunction();
    };

} /* end namespace store */
} /* end namespace ddj */

#endif /* defined(DDJ_Store__DDJ_StoreBuffer_h) */
