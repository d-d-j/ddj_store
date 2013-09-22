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
#include "../Task/StoreTask.h"

namespace ddj {
namespace store {

class StoreController
{
    /* TYPEDEFS */
    typedef std::shared_ptr<StoreBuffer> StoreBuffer_Pointer;
    typedef std::pair<tag_type, StoreBuffer_Pointer> store_hash_value_type;

    /* FIELDS */
    private:
        __gnu_cxx::hash_map<tag_type, StoreBuffer_Pointer>* _buffers;
        boost::ptr_vector<StoreTask> _tasks;
        boost::thread _notificationThread;
        boost::condition_variable _notificationCond;
        boost::mutex _notificationMutex;
    public:
        StoreController();
        ~StoreController();

        bool InsertValue(storeElement* element);
        bool InsertValue(int series, tag_type tag, ullint time, store_value_type value);

    private:

        void startNotificationThread();
        void stopNotificationThread();
        void notificationThreadFunction();
};

} /* end namespace store */
} /* end namespace ddj */

#endif /* defined(DDJ_Store_DDJ_StoreController_h) */
