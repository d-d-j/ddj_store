/*
 * BTreeMonitor.cpp
 *
 *  Created on: Aug 24, 2013
 *      Author: parallels
 */

#include "BTreeMonitor.h"

namespace ddj {
namespace store {

BTreeMonitor::BTreeMonitor(tree_pointer tree)
{
	this->_bufferInfoTree = tree;
	this->startThread();
}

BTreeMonitor::~BTreeMonitor()
{
	this->stopThread();
}

void BTreeMonitor::threadFunction()
{
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
		h_LogThreadDebug("Tree inserter doing JOB.");
		if(!this->insertToTree(this->_infoElementToInsert)) h_LogThreadError("Tree inserter failed inserting infoElement");
		h_LogThreadDebug("Tree inserter has done his JOB.");
		this->_treeBufferThreadBusy = 0;
		this->_condTreeBufferFree.notify_one();
	}
}

void BTreeMonitor::startThread()
{
	/* START tree inserter job */
	this->_threadBufferTree = new boost::thread(boost::bind(&BTreeMonitor::threadFunction, this));	// Create thread which will be inserting infoElements to BTree
	boost::mutex::scoped_lock lock(this->_mutexSynchronization);	// Lock the synchronization mutex
	this->_condSynchronization.wait(lock, boost::lambda::var(this->_threadBufferTreeStop));		// Wait for tree thread to initialize
	h_LogThreadDebug("Tree inserter thread started!");
}

void BTreeMonitor::stopThread()
{
	/* STOP tree inserter job */
	h_LogThreadDebug("Blocking info tree mutex.");
	this->_mutexBufferInfoTree.lock();
	h_LogThreadDebug("Mutex blocked");

	this->_threadBufferTreeStop = false;	//tree buffer job is not working
	this->_condBufferInfoTree.notify_one();

	this->_mutexBufferInfoTree.unlock();

	h_LogThreadDebug("Joining tree inserter thread.");
	this->_threadBufferTree->join();	//tree buffer job (Thread) should be joined
	/* LOG */
	h_LogThreadDebug("Tree inserter thread joined.");
}

bool BTreeMonitor::insertToTree(infoElement* element)
{
	try
	{
		if(this->_bufferInfoTree == NULL) throw;
		this->_bufferInfoTree->insert(element->startTime, element->startValue);
		this->_bufferInfoTree->insert(element->endTime, element->endValue);
		h_LogThreadDebug("Inserted infoElement to B+Tree.");
		return true;
	}
	catch(...)
	{
	h_LogThreadError("Inserting infoElement to B+Tree FAILED!!");
	return false;
	}
}

void BTreeMonitor::Insert(infoElement* element)
{
	boost::mutex::scoped_lock lock(this->_mutexBufferInfoTree);
	this->_condTreeBufferFree.wait(lock, !boost::lambda::var(this->_treeBufferThreadBusy));
	this->_treeBufferThreadBusy = 1;
	pantheios::log_DEBUG("Passing info element to tree inserter thread...");
	this->_infoElementToInsert = element;
	lock.unlock();
	this->_condBufferInfoTree.notify_one();
	pantheios::log_DEBUG("Info element passed to tree inserter thread!");
}

} /* namespace store */
} /* namespace ddj */
