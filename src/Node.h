/*
 * Node.h
 *
 *  Created on: 30-09-2013
 *      Author: ghashd
 */

#ifndef NODE_H_
#define NODE_H_

#include "Network/Client.h"
#include "Task/StoreTaskMonitor.h"
#include "Task/TaskType.h"
#include "Task/StoreTask.h"
#include "Task/TaskRequest.h"
#include "Task/TaskResult.h"
#include "Store/StoreController.h"
#include "Store/storeElement.h"
#include "Helpers/Logger.h"

namespace ddj
{
	class Node
	{
	private:
		/* TYPEFEFS */
		typedef boost::shared_ptr<store::StoreController> StoreController_Pointer;

		/* LOGGER */
		Logger _logger = Logger::getRoot();

        /* NETWORK */
        Client* _client;
        boost::signals2::signal<void (taskRequest)> _requestSignal;

        /* STORE CONTROLLER */
        int _cudaDevicesCount;
        boost::unordered_map<int, StoreController_Pointer> _controllers;

        /* TASK */
    	store::StoreTaskMonitor* _storeTaskMonitor;
        boost::thread* _taskThread;
        boost::condition_variable _taskCond;
        boost::mutex _taskMutex;
        boost::barrier* _taskBarrier;
	public:
		Node();
		virtual ~Node();
        void CreateTask(taskRequest request);

	private:
        void taskThreadFunction();
	};

} /* namespace ddj */
#endif /* NODE_H_ */
