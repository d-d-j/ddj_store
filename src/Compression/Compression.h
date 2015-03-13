#ifndef COMPRESSION_H_
#define COMPRESSION_H_

#include "CompressionLightweight.cuh"

namespace ddj {
namespace compression {

	using namespace ddj::store;

	/**
	 * @class Compression
	 *
	 */
	class Compression
	{
	public:
		size_t CompressTrunk(storeElement* elements, size_t size, void** result, cudaStream_t stream);
		size_t DecompressTrunk(void* data, size_t size, storeElement** result);
	};

} /* namespace compression */
} /* namespace ddj */
#endif /* COMPRESSION_H_ */
