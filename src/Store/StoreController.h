/*
 *  StoreController.h
 *  StoreController
 *
 *  Created by Karol Dzitkowski on 27.07.2013.
 *  Copyright (c) 2013 Karol Dzitkowski. All rights reserved.
 *
 *      NAZEWNICTWO
 * 1. nazwy klas:  CamelStyle z dużej litery np. StoreController
 * 2. nazwy struktur camelStyle z małej litery np. storeElement
 * 3. nazwy pól prywatnych camelStyle z małej litery z podkreśleniem _backBuffer
 * 4. nazwy pól publicznych i zmiennych globalnych słowa rozdzielamy _ i z małych liter np. memory_available
 * 5. define z dużych liter i rozdzielamy _ np. BUFFER_SIZE
 * 6. nazwy metod publicznych z dużej litery CamelStyle np. InsertValue() oraz parametry funkcji z małych liter camelStyle np. InsertValue(int valToInsert);
 * 7. nazwy metod prywatnych z małej litery camelStyle
 * 8. nazwy funkcji "prywatnych" w plikach cpp z małej litery z _ czyli, insert_value(int val_to_insert);
 * 9. nazwy funkcji globalnych czyli w plikach .h najczęściej inline h_InsertValue() dla funkcji na CPU g_InsertValue() dla funkcji na GPU
 */

#ifndef DDJ_Store_DDJ_StoreController_h
#define DDJ_Store_DDJ_StoreController_h

#include "StoreBuffer.h"
#include "../Network/Client.h"
#include "../Task/TaskType.h"
#include "../Task/StoreTask.h"
#include "../Task/StoreTaskMonitor.h"
#include "../GpuUpload/GpuUploadMonitor.h"
#include "../Task/TaskRequest.h"

namespace ddj {
namespace store {

class StoreController
{
    /* TYPEDEFS */
    typedef boost::function<void (StoreTask_Pointer task)> taskFunc;
    typedef boost::shared_ptr<StoreBuffer> StoreBuffer_Pointer;

    /* FIELDS */
    private:
    	GpuUploadMonitor _gpuUploadMonitor;
    	StoreTaskMonitor* _storeTaskMonitor;
    	boost::unordered_map<tag_type, StoreBuffer_Pointer>* _buffers;
        boost::unordered_map<int, taskFunc> _taskFunctions;

        /* TASK */
        boost::thread* _taskThread;
        boost::condition_variable _taskCond;
        boost::mutex _taskMutex;
        boost::barrier* _taskBarrier;

        /* NETWORK */
        Client* _client;
        boost::signals2::signal<void (taskRequest)> _requestSignal;

	/* METHODS */
    public:
        StoreController();
        virtual ~StoreController();

        void CreateTask(taskRequest request);
    private:
        void taskThreadFunction();
        void populateTaskFunctions();

	/* TASK FUNCTIONS */
    private:
        void insertTask(StoreTask_Pointer task);
};

} /* end namespace store */
} /* end namespace ddj */

#endif /* defined(DDJ_Store_DDJ_StoreController_h) */
