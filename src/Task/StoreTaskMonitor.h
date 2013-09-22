/*
 * StoreTaskMonitor.h
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#ifndef STORETASKMONITOR_H_
#define STORETASKMONITOR_H_

namespace ddj {
namespace store {

class StoreTaskMonitor
{
	/* FIELDS */
	public:
    	boost::ptr_vector<StoreTask> _tasks;
    	boost::condition_variable* _condResponseReady;

	/* METHODS */
	public:
		StoreTaskMonitor(boost::condition_variable* condResponseReady);
		virtual ~StoreTaskMonitor();
		StoreTask* AddTask(int taskId, TaskType type, void* taskData, int dataSize);
};

} /* namespace store */
} /* namespace ddj */
#endif /* STORETASKMONITOR_H_ */
