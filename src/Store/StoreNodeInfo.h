/*
 * StoreNodeInfo.h
 *
 *  Created on: Dec 15, 2013
 *      Author: dud
 */

#ifndef STORENODEINFO_H_
#define STORENODEINFO_H_

#include <string>
#include <sstream>

namespace ddj
{
namespace store
{
class StoreNodeInfo
{
	int32_t _memTotal;
	int32_t _memFree;
	int32_t _gpuMemTotal;
	int32_t _gpuMemFree;
public:
	StoreNodeInfo();
	StoreNodeInfo(int32_t memTotal, int32_t memFree, int32_t gpuMemTotal, int32_t gpuMemFree)
		: _memTotal(memTotal), _memFree(memFree), _gpuMemTotal(gpuMemTotal), _gpuMemFree(gpuMemFree) {}
	std::string toString()
	{
		 std::ostringstream stream;
	     stream << "RAM: "<<_memFree<<"/"<<_memTotal<<"\t GPU: "<<_gpuMemFree<<"/"<<_gpuMemTotal;
	     return  stream.str();
	}
};
}
}
#endif /* STORENODEINFO_H_ */
