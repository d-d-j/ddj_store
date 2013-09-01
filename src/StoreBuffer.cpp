/*
 * StoreBuffer.cpp
 *
 *  Created on: Aug 10, 2013
 *      Author: Karol Dzitkowski
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

#include "StoreBuffer.h"

using namespace ddj::store;

StoreBuffer::StoreBuffer(tag_type tag)
{
	if(typeid(tag_type) == typeid(int))
		pantheios::log_DEBUG("StoreBuffer with [Tag = ", pantheios::integer(tag), "] is being created");

	this->_tag = tag;
	this->_bufferElementsCount = 0;
	this->_bufferInfoTree = new tree();
	this->_bufferInfoTreeMonitor = new BTreeMonitor(this->_bufferInfoTree);
}

StoreBuffer::~StoreBuffer()
{
	if(typeid(tag_type) == typeid(int))
		pantheios::log_DEBUG(PSTR("StoreBuffer [Tag = "), pantheios::integer(this->_tag), PSTR("] is being freed"));

	delete this->_bufferInfoTreeMonitor;
	delete this->_bufferInfoTree;

	if(typeid(tag_type) == typeid(int))
			pantheios::log_DEBUG(PSTR("StoreBuffer [Tag = "), pantheios::integer(this->_tag), PSTR("] has been freed"));
}

infoElement* StoreBuffer::insertToBuffer(storeElement* element)
{
	this->_buffer[_bufferElementsCount] = *element;
	this->_bufferElementsCount++;
	if(this->_bufferElementsCount == STORE_BUFFER_SIZE)
	{
		this->switchBuffers();
		return new infoElement(element->tag, this->_buffer[0].time, element->time, this->_buffer[0].value, element->value);
	}
	return NULL;
}

void StoreBuffer::switchBuffers()
{
	this->_bufferElementsCount = 0;
	this->_buffer.swap(this->_backBuffer);
}

bool StoreBuffer::InsertElement(storeElement* element)
{
	// Firstly we want to insert received element to buffer, if buffer is full it is switched and info_element is returned
	infoElement* result = this->insertToBuffer(element);
	if(result != NULL)
	{
		// Buffers are now switched and it is time to upload back_buffer content to GPU memory
		//
		//	TODO: Uploading back_buffer to GPU memory
		//

		// After uploading back_buffer to GPU memory and result->startVAlue and result->endValue have been set
		// There is time to Insert information about TRUNKS to buffer_tree

		this->_bufferInfoTreeMonitor->Insert(result);
		return true;
	}
	return false;
}

void StoreBuffer::Flush()
{
	// Switch the buffers
	this->switchBuffers();
	// Synchronized upload of back_buffer to GPU memory
	//
	// TODO: Synchronized upload to GPU memory
	//
	// OR waiting for memory upload to end
}
