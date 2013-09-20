/*
 *  StoreController.cpp
 *  StoreController
 *
 *  Created by Karol Dzitkowski on 27.07.2013.
 *  Copyright (c) 2013 Karol Dzitkowski. All rights reserved.
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

#include "StoreController.h"

namespace ddj {
namespace store {

	StoreController::StoreController()
	{
		pantheios::log_DEBUG("Initializing new StoreController - creating StoreBuffer hash map");
		_buffers = new __gnu_cxx::hash_map<tag_type, StoreBuffer_Pointer>();
	}

	StoreController::~StoreController()
	{
		pantheios::log_DEBUG("Removing StoreController - delete StoreBuffer hash map");
		delete _buffers;
	}

	bool StoreController::InsertValue(storeElement* element)
	{
		if(_buffers->count(element->tag) == 0)
		{
			std::shared_ptr<StoreBuffer> p(new StoreBuffer(element->tag));
			_buffers->insert(store_hash_value_type( element->tag, p));
		}
		return (*_buffers)[element->tag]->InsertElement(element);
	}

	bool StoreController::InsertValue(int series, tag_type tag, ullint time, store_value_type value)
	{
		return this->InsertValue(new storeElement(series, tag, time, value));
	}

	void StoreController::startNotificationThread()
	{

	}

	void StoreController::stopNotificationThread()
	{

	}

	void StoreController::notificationThreadFunction()
	{

	}

}
}
