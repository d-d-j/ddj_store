/*
 * StoreNodeInfo.h
 *
 *  Created on: Dec 15, 2013
 *      Author: dud
 */

#ifndef STORENODEINFO_H_
#define STORENODEINFO_H_

#include <stdio.h>
#include <stdlib.h>

namespace ddj
{
namespace store
{
class StoreNodeInfo
{
	int _memTotal;
	int _memFree;
	size_t _gpuMemTotal;
	size_t _gpuMemFree;
public:
	StoreNodeInfo();
	StoreNodeInfo(int memTotal, int memFree, size_t gpuMemTotal, size_t gpuMemFree);
	virtual ~StoreNodeInfo();
};
}
}
#endif /* STORENODEINFO_H_ */
