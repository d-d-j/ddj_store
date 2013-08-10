//
//  DDJ_StoreController.h
//  DDJ_StoreController
//
//  Created by Karol Dzitkowski on 27.07.2013.
//  Copyright (c) 2013 Karol Dzitkowski. All rights reserved.
//

#ifndef DDJ_Store_DDJ_StoreController_h
#define DDJ_Store_DDJ_StoreController_h

#include "DDJ_StoreBuffer.h"

namespace ddj {
namespace store {

class StoreController
{
    /* TYPEDEFS */
    typedef StoreBuffer* StoreBuffer_Pointer;
    typedef std::pair<tag_type, StoreBuffer_Pointer> store_hash_value_type;

    /* FIELDS */
    private:
        __gnu_cxx::hash_map<tag_type, StoreBuffer_Pointer>* _buffers;

    public:
        StoreController();
        ~StoreController();
        bool InsertValue(storeElement* element);
        bool InsertValue(int series, tag_type tag, ullint time, store_value_type value);
};

} /* end namespace store */
} /* end namespace ddj */

#endif /* defined(DDJ_Store_DDJ_StoreController_h) */
