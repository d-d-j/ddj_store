#ifndef STORENODEINFO_H_
#define STORENODEINFO_H_

#include <string>
#include <sstream>

namespace ddj {
namespace store {

	struct storeNodeInfo
	{
	public:
		int32_t gpu_id;
		int32_t mem_total;
		int32_t mem_free;
		int32_t gpu_mem_total;
		int32_t gpu_mem_free;

		storeNodeInfo();
		storeNodeInfo(int32_t gpuId, int32_t memTotal, int32_t memFree, int32_t gpuMemTotal, int32_t gpuMemFree)
			:gpu_id(gpuId), mem_total(memTotal), mem_free(memFree), gpu_mem_total(gpuMemTotal), gpu_mem_free(gpuMemFree) {}

		bool operator== (const storeNodeInfo& rhs) const
		{
			if(mem_total == rhs.mem_total && mem_free == rhs.mem_free &&
					gpu_mem_total == rhs.gpu_mem_total && gpu_mem_free == rhs.gpu_mem_free)
			{
				return true;
			}
			return false;
		}

		std::string toString()
		{
			 std::ostringstream stream;
			 stream << "RAM: "<<mem_free<<"/"<<mem_total<<"\t GPU: "<<gpu_mem_free<<"/"<<gpu_mem_total;
			 return  stream.str();
		}
	};

} /* namespace store */
} /* namespace ddj */
#endif /* STORENODEINFO_H_ */
