/*
 * BTreeMonitor.cpp
 *
 *  Created on: Aug 24, 2013
 *      Author: parallels
 */

#include "BTreeMonitor.h"

namespace ddj {
namespace store {

	BTreeMonitor::BTreeMonitor(tag_type tag)
	{
		this->_tag = tag;
		this->_bufferInfoTree = new tree();
	}

	BTreeMonitor::~BTreeMonitor()
	{
		delete this->_bufferInfoTree;
	}

	void BTreeMonitor::Insert(infoElement* element)
	{
		h_LogThreadWithTagDebug("Attempt to lock BTreeMonitor mutex", this->_tag);
		boost::lock_guard<boost::mutex> guard(this->_mutex);
		h_LogThreadWithTagDebug("BTreeMonitor mutex locked", this->_tag);
		this->insertToTree(element);
	}

	void BTreeMonitor::insertToTree(infoElement* element)
	{
		try
		{
			if(this->_bufferInfoTree == NULL) throw;
			this->_bufferInfoTree->insert(element->startTime, element->startValue);
			this->_bufferInfoTree->insert(element->endTime, element->endValue);
			h_LogThreadDebug("Insert an infoElement to B+Tree [success]");
		}
		catch(...)
		{
			h_LogThreadError("Insert an infoElement to B+Tree [Failure]");
		}
	}

} /* namespace store */
} /* namespace ddj */
