/*
 * BTreeMonitor.cpp
 *
 *  Created on: Aug 24, 2013
 *      Author: parallels
 */

#include "BTreeMonitor.h"

namespace ddj {
namespace btree {

	BTreeMonitor::BTreeMonitor(metric_type metric)
	{
		LOG4CPLUS_DEBUG_FMT(this->_logger, "Btree monitor [metric=%d] constructor [BEGIN]", metric);

		this->_metric = metric;
		this->_bufferInfoTree = new tree();

		LOG4CPLUS_DEBUG_FMT(this->_logger, "Btree monitor [metric=%d] constructor [END]", metric);
	}

	BTreeMonitor::~BTreeMonitor()
	{
		delete this->_bufferInfoTree;
	}

	void BTreeMonitor::Insert(store::storeTrunkInfo* element)
	{
		boost::lock_guard<boost::mutex> guard(this->_mutex);
		this->insertToTree(element);
	}

	void BTreeMonitor::insertToTree(store::storeTrunkInfo* element)
	{
		try
		{
			this->_bufferInfoTree->insert(element->startTime, element->startValue);
			this->_bufferInfoTree->insert(element->endTime, element->endValue);
			LOG4CPLUS_DEBUG_FMT(this->_logger, "BTreeMonitor - insert element to b+tree: {tag=%d, startT=%llu, endT=%llu, startV=%d, endV=%d}",
					element->metric, element->startTime, element->endTime, element->startValue, element->endValue);
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
} /* namespace btree */
