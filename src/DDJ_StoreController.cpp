//
//  DDJ_StoreController.cpp
//  DDJ_StoreController
//
//  Created by Karol Dzitkowski on 27.07.2013.
//  Copyright (c) 2013 Karol Dzitkowski. All rights reserved.
//

#include "DDJ_StoreController.h"

namespace ddj {
namespace store {



StoreController::StoreController()
{
    pantheios::log_DEBUG("Initializing new StoreController - creating StoreBuffer hash map");
    _buffers = new __gnu_cxx::hash_map<tag_type, StoreBuffer_Pointer>();
}

StoreController::~StoreController()
{
	pantheios::log_DEBUG("Removing StoreController - delete StoreBuffer hash map");
	delete _buffers;
}

bool StoreController::InsertValue(storeElement* element)
{
	StoreBuffer_Pointer newBuffer;
	if(_buffers->count(element->tag) == 0)
	{
		_buffers->insert(store_hash_value_type( element->tag, new StoreBuffer(element->tag)) );
	}
	return (*_buffers)[element->tag]->InsertElement(element);
}

bool StoreController::InsertValue(int series, tag_type tag, ullint time, store_value_type value)
{
    return this->InsertValue(new storeElement(series, tag, time, value));
}




}
}
