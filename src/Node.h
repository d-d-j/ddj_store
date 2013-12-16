/*
 * Node.h
 *
 *  Created on: 30-09-2013
 *      Author: ghashd
 */

#ifndef NODE_H_
#define NODE_H_

#include "Task/TaskType.h"
#include "Task/TaskRequest.h"
#include "Task/TaskResult.h"
#include "Task/TaskMonitor.h"
#include "Store/StoreController.h"
#include "Store/StoreElement.h"
#include "Cuda/CudaCommons.h"
#include "Core/Logger.h"
#include "Core/Config.h"
#include "Network/NetworkClient.h"


namespace ddj
{
	class Node : public boost::noncopyable
	{
	private:
		/* TYPEFEFS */
		typedef boost::shared_ptr<store::StoreController> StoreController_Pointer;

		/* LOGGER & CONFIG & CUDA_COMMONS */
		store::CudaCommons _cudaCommons;
		Logger _logger = Logger::getRoot();
		Config* _config = Config::GetInstance();

        /* NETWORK */
        network::NetworkClient* _client;
        boost::signals2::signal<void (task::taskRequest)> _requestSignal;

        /* STORE CONTROLLER */
        int _cudaDevicesCount;
        boost::unordered_map<int, StoreController_Pointer> _controllers;

        /* TASK */
    	task::TaskMonitor* _taskMonitor;
        boost::thread* _taskThread;
        boost::condition_variable _taskCond;
        boost::mutex _taskMutex;
        boost::barrier* _taskBarrier;
	public:
		Node();
		virtual ~Node();
        void CreateTask(task::taskRequest request);

	private:
        void taskThreadFunction();
	};

} /* namespace ddj */
#endif /* NODE_H_ */
