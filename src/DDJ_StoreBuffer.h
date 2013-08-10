/*
 *  DDJ_StoreBuffer.h
 *  DDJ_StoreController
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

#include "DDJ_StoreIncludes.h"
#include "btree.h"

#define STORE_BUFFER_SIZE 512

namespace ddj {
namespace store {
    
	/* TYPEDEFS */
	typedef unsigned long long int ullint;
	typedef stx::btree<ullint, int> tree;
	typedef tree* tree_pointer;
    typedef int tag_type;
    typedef float store_value_type;
    typedef float info_value_type;

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

    typedef struct infoElement
	{
		public:
    		/* FIELDS */
			tag_type tag;
			ullint startTime;
			ullint endTime;
			info_value_type startValue;
			info_value_type endValue;
			/* CONSTRUCTORS */
			infoElement(){ tag = 0; startTime = 0; endTime = 0; startValue = 0; endValue = 0; }
			infoElement(tag_type _tag, ullint _startTime, ullint _endTime, info_value_type _startValue, info_value_type _endValue)
			: tag(_tag), startTime(_startTime), endTime(_endTime), startValue(_startValue), endValue(_endValue) {}
			~infoElement() {}
	} infoElement;

    /* CLASSES */
    class StoreBuffer
    {
		/* FIELDS */
		private:
			/* BASIC */
			int _tag;
			/* THREADS */
			boost::thread* _threadForTree;

			 /* TREE */
			 tree_pointer _bufferTree;

			 /* BUFFER */
			 int _bufferElementsCount;
			 bool _backBufferEmpty;
			 boost::array<storeElement, STORE_BUFFER_SIZE> _buffer;
			 boost::array<storeElement, STORE_BUFFER_SIZE> _backBuffer;

		/* METHODS */
		public:
			 StoreBuffer(tag_type tag);
			~StoreBuffer();
			bool InsertElement(storeElement* element);
			void Flush();
			void Start();
			void Stop();
		private:
			/* BUFFER METHODS */
			infoElement* insertToBuffer(storeElement* element);
			void switchBuffers();

			/* TREE METHODS */
			void insertToTree(infoElement* element);
			void treeInserterJob();
    };

} /* end namespace store */
} /* end namespace ddj */

#endif /* defined(DDJ_Store__DDJ_StoreBuffer_h) */
