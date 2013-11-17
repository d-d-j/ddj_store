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
		LOG4CPLUS_DEBUG_FMT(this->_logger, "Btree monitor [tag=%d] constructor [BEGIN]", tag);

		this->_tag = tag;
		this->_bufferInfoTree = new tree();

		LOG4CPLUS_DEBUG_FMT(this->_logger, "Btree monitor [tag=%d] constructor [END]", tag);
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
			this->_bufferInfoTree->insert(element->startTime, element->startValue);
			this->_bufferInfoTree->insert(element->endTime, element->endValue);
			LOG4CPLUS_DEBUG_FMT(this->_logger, "BTreeMonitor - insert element to b+tree: {tag=%d, startT=%llu, endT=%llu, startV=%d, endV=%d}",
					element->tag, element->startTime, element->endTime, element->startValue, element->endValue);
		}
		catch(std::exception& ex)
		{
			LOG4CPLUS_ERROR_FMT(this->_logger, "Inserting to B+Tree error with exception - [%s] [FAILED]", ex.what());
		}
		catch(...)
		{
			LOG4CPLUS_FATAL(this->_logger, LOG4CPLUS_TEXT("Inserting to B+Tree error [FAILED]"));
		}
	}

} /* namespace store */
} /* namespace ddj */
