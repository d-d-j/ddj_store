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
#include "../GpuUpload/GpuUploadMonitor.h"
#include "../Task/StoreTask.h"
#include "../CUDA/CudaController.h"
#include "../Store/storeSettings.h"
#include "../Query/QueryMonitor.h"
#include "../Helpers/Config.h"
#include "../Helpers/Logger.h"


namespace ddj {
namespace store {

class StoreController
{
    /* TYPEDEFS */
    typedef boost::function<void (StoreTask_Pointer task)> taskFunc;
    typedef boost::shared_ptr<StoreBuffer> StoreBuffer_Pointer;
    typedef boost::unordered_map<metric_type, StoreBuffer_Pointer> Buffers_Map;

    /* FIELDS */
    private:
    	int _gpuDeviceId;
    	GpuUploadMonitor* _gpuUploadMonitor;
    	QueryMonitor* _queryMonitor;
    	CudaController* _cudaController;

    	Buffers_Map* _buffers;
        boost::unordered_map<int, taskFunc> _taskFunctions;

        /* LOGGER & CONFIG */
		Logger _logger = Logger::getRoot();
		Config* _config = Config::GetInstance();

	/* METHODS */
    public:
        StoreController(int gpuDeviceId);
        virtual ~StoreController();
        void ExecuteTask(StoreTask_Pointer task);
    private:
        void populateTaskFunctions();

	/* TASK FUNCTIONS */
    private:
        void insertTask(StoreTask_Pointer task);
        void selectAllTask(StoreTask_Pointer task);
        void flushTask(StoreTask_Pointer task);
};

} /* end namespace store */
} /* end namespace ddj */

#endif /* defined(DDJ_Store_DDJ_StoreController_h) */
