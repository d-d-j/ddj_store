/*
 * BTreeMonitor.h
 *
 *  Created on: Aug 24, 2013
 *      Author: Karol Dzitkowski
 */

#ifndef BTREEMONITOR_H_
#define BTREEMONITOR_H_

#include "DDJ_StoreIncludes.h"

namespace ddj {
namespace store {

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

class BTreeMonitor
{
private:
	boost::thread* _threadBufferTree;

	boost::condition_variable _condBufferInfoTree;
	boost::condition_variable _condTreeBufferFree;

	boost::mutex _mutexSynchronization;
	boost::mutex _mutexBufferInfoTree;

	boost::condition_variable _condSynchronization;

	volatile sig_atomic_t _threadBufferTreeStop;
	volatile sig_atomic_t _treeBufferThreadBusy;

	//TODO: Change _infoElementToInsert to queue of infoElements
	infoElement* _infoElementToInsert;
	tree_pointer _bufferInfoTree;

public:
	BTreeMonitor();
	BTreeMonitor(tree_pointer tree);
	virtual ~BTreeMonitor();
	void Insert(infoElement* element);
private:
	void threadFunction();
	bool insertToTree(infoElement* element);
	void startThread();
	void stopThread();
};





















} /* namespace store */
} /* namespace ddj */
#endif /* BTREEMONITOR_H_ */
