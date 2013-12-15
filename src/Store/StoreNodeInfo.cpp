/*
 * StoreNodeInfo.cpp
 *
 *  Created on: Dec 15, 2013
 *      Author: dud
 */


#include "StoreNodeInfo.h"
namespace ddj
{
namespace store
{
StoreNodeInfo::StoreNodeInfo(int memTotal, int memFree, size_t gpuMemTotal,
		size_t gpuMemFree)
{
	_memTotal = memTotal;
	_memFree = memFree;
	_gpuMemFree = gpuMemFree;
	_gpuMemTotal = gpuMemTotal;
}

StoreNodeInfo::StoreNodeInfo()
{
// TODO Auto-generated constructor stub

}

StoreNodeInfo::~StoreNodeInfo()
{
// TODO Auto-generated destructor stub
}
}
}
