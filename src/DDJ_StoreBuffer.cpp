/*
 * DDJ_StoreBuffer.cpp
 *
 *  Created on: Aug 10, 2013
 *      Author: parallels
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

#include "DDJ_StoreBuffer.h"

using namespace ddj::store;

////////////////////////////////////////////////////////////////
///////////// HELP FUNCTIONS DECLARATIONS //////////////////////
////////////////////////////////////////////////////////////////

void insert_one_item_to_tree(tree_pointer tree, tag_type tag, ullint time, info_value_type value);

////////////////////////////////////////////////////////////////
///////////// STORE BUFFER METHODS IMPLEMENTATIONS /////////////
////////////////////////////////////////////////////////////////

StoreBuffer::StoreBuffer(tag_type tag)
{
	if(typeid(tag_type) == typeid(int))
		pantheios::log_DEBUG("StoreBuffer with [Tag = ", pantheios::integer(tag), "] is being created");

	this->_tag = tag;
	this->_bufferElementsCount = 0;
	this->_bufferInfoTree = new tree();
	this->Start();
}

StoreBuffer::~StoreBuffer()
{
	if(typeid(tag_type) == typeid(int))
		pantheios::log_DEBUG(PSTR("StoreBuffer [Tag = "), pantheios::integer(this->_tag), PSTR("] is being freed"));

	this->Stop();
	delete this->_bufferInfoTree;
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

void StoreBuffer::insertToTree(infoElement* element)
{
	insert_one_item_to_tree(this->_bufferInfoTree, element->tag, element->startTime, element->startValue);
	insert_one_item_to_tree(this->_bufferInfoTree, element->tag, element->endTime, element->endValue);
}

void insert_one_item_to_tree(tree_pointer tree, tag_type tag, ullint time, info_value_type value)
{
	if(tree != NULL)
	{
		tree->insert(time, value);
	}
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
		boost::mutex::scoped_lock lock(this->_mutexBufferInfoTree);
		this->_infoElementToInsert = result;
		this->_condBufferInfoTree.notify_one();
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

void StoreBuffer::Start()
{
	/* START tree inserter job */
	this->_threadBufferTree = new boost::thread(boost::bind(&StoreBuffer::treeInserterJob, this));	// Create thread for bufferInfoTree

	{
		boost::mutex::scoped_lock lock(this->_mutexSynchronization);	// Lock the synchronization mutex
		this->_condSynchronization.wait(lock, boost::lambda::var(this->_threadBufferTreeStop));		// Wait for tree thread to initialize
	}

	/* LOG */
	pantheios::log_DEBUG(
			"Thread for tree with [Tag = ",
			pantheios::integer(this->_tag),
			"] was started as [Thread id = ",
			boost::lexical_cast<std::string>(this->_threadBufferTree->get_id()),
			"]");
}

void StoreBuffer::Stop()
{
	/* STOP tree inserter job */
	h_LogThreadDebug("Blocking buffer mutex.");

	this->_mutexBufferInfoTree.lock();
	h_LogThreadDebug("Mutex blocked");
	this->_threadBufferTreeStop = false;	//tree buffer job is not working
	this->_condBufferInfoTree.notify_one();
	this->_mutexBufferInfoTree.unlock();

	h_LogThreadDebug("Joining tree inserter thread.");
	this->_threadBufferTree->join();	//tree buffer job (Thread) should be joined
	/* LOG */
	pantheios::log_DEBUG(
			"Thread for tree with [Tag =",
			pantheios::integer(this->_tag),
			"] was joined as [Thread id = ",
			boost::lexical_cast<std::string>(this->_threadBufferTree->get_id()),
			"]");
}

void StoreBuffer::treeInserterJob()
{
	/* LOG */
	h_LogThreadDebug("Thread started!");
	boost::mutex::scoped_lock lock(this->_mutexBufferInfoTree);
	h_LogThreadDebug("Tree inserter locked his mutex.");
	//Creating unique_lock
	{
		boost::mutex::scoped_lock synclock(this->_mutexSynchronization);
		// Awake StoreBuffer main thread that this thread was properly started
		this->_threadBufferTreeStop = true;
		this->_condSynchronization.notify_one();
	}
	this->_treeBufferThreadBusy = 0;
	this->_condTreeBufferFree.notify_one();
	// Thread is sleeping until buffer is full and was switched
	while(this->_threadBufferTreeStop)
	{
		h_LogThreadDebug("Tree inserter begin waiting.");
		this->_condBufferInfoTree.wait(lock);

		if(this->_threadBufferTreeStop == false)
		{
			h_LogThreadDebug("Tree inserter is exiting.");
			break;
		}
		this->_tag++;
		h_LogThreadDebug("Tree inserter doing JOB.");
		boost::this_thread::sleep(boost::posix_time::milliseconds(200));
		h_LogThreadDebug("Tree inserter has done his JOB.");
		this->_treeBufferThreadBusy = 0;
		this->_condTreeBufferFree.notify_one();
	}
}

void StoreBuffer::TESTME()
{
	boost::mutex::scoped_lock lock(this->_mutexBufferInfoTree);
	this->_condTreeBufferFree.wait(lock, !boost::lambda::var(this->_treeBufferThreadBusy));
	this->_treeBufferThreadBusy = 1;
	pantheios::log_DEBUG("TEST started");
	this->_infoElementToInsert = new infoElement(5, 10, 15, 1, 20);
	lock.unlock();
	this->_condBufferInfoTree.notify_one();
	//boost::this_thread::sleep(boost::posix_time::milliseconds(2));
	pantheios::log_DEBUG("exiting TEST function");
}
