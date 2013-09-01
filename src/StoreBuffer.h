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

#include "BTreeMonitor.h"
#include "GpuUploaderMonitor.h"

namespace ddj {
namespace store {

    /* CLASSES */
    class StoreBuffer
    {
		/* FIELDS */
		private:
			/* BASIC */
			int _tag;

			/* TREE */
			tree_pointer _bufferInfoTree;
			BTreeMonitor* _bufferInfoTreeMonitor;

			/* BUFFER */
			GpuUploaderMonitor* _gpuUploader;
			int _bufferElementsCount;
			boost::array<storeElement, STORE_BUFFER_SIZE> _buffer;
			boost::array<storeElement, STORE_BUFFER_SIZE> _backBuffer;

		/* METHODS */
		public:
			StoreBuffer(tag_type tag);
			virtual ~StoreBuffer();
			bool InsertElement(storeElement* element);
			void Flush();
		private:
			/* BUFFER METHODS */
			infoElement* insertToBuffer(storeElement* element);
			void switchBuffers();
    };

} /* end namespace store */
} /* end namespace ddj */

#endif /* defined(DDJ_Store__DDJ_StoreBuffer_h) */
