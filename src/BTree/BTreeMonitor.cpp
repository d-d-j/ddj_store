#include "BTreeMonitor.h"
#include <boost/foreach.hpp>

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
		try
		{
			this->_bufferInfoTree->insert(
					ullintPair{element->startTime,element->endTime},
					ullintPair{element->startValue, element->endValue});

			LOG4CPLUS_DEBUG_FMT(this->_logger, "BTreeMonitor - insert element to b+tree: {tag=%d, startT=%llu, endT=%llu, startV=%llu, endV=%llu}",
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

	boost::container::vector<ullintPair>* BTreeMonitor::SelectAll()
	{
		boost::container::vector<ullintPair>* result = new boost::container::vector<ullintPair>();

		auto it = this->_bufferInfoTree->begin();
		for(; it != this->_bufferInfoTree->end(); it++)
			result->push_back(it->second);

		return result;
	}

	bool BTreeMonitor::isIntersecting(ullintPair A, ullintPair B)
	{
		if(A.first > B.first && A.first < B.second) return true;
		if(A.second > B.first && A.second < B.second) return true;
		if(B.first > A.first && B.first < A.second) return true;
		if(B.second > A.first && B.second < A.second) return true;
		return false;
	}

	boost::container::vector<ullintPair>* BTreeMonitor::Select(boost::container::vector<ullintPair> timePeriods)
	{
		boost::container::vector<ullintPair>* result = new boost::container::vector<ullintPair>();

		BOOST_FOREACH(ullintPair &tp, timePeriods)
		{
			auto it = this->_bufferInfoTree->lower_bound(tp);
			it--;
			if(isIntersecting(it->first, tp))
			{
				result->push_back(it->second);
			}
			it++;
			while(it->first.first < tp.second && it != this->_bufferInfoTree->end())
			{
				result->push_back(it->second);
				it++;
			}
		}

		return result;
	}

} /* namespace store */
} /* namespace btree */
