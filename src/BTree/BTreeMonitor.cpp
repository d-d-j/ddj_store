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
		boost::lock_guard<boost::mutex> guard(this->_mutex);
		this->insertToTree(element);
	}

	void BTreeMonitor::insertToTree(infoElement* element)
	{
		try
		{
			if(this->_bufferInfoTree == NULL) throw;
			this->_bufferInfoTree->insert(element->startTime, element->startValue);
			this->_bufferInfoTree->insert(element->endTime, element->endValue);
		}
		catch(...)
		{
		}
	}

} /* namespace store */
} /* namespace ddj */
