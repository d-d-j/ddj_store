#include "Compression.h"

namespace ddj {
namespace compression {

size_t Compression::CompressTrunk(storeElement* elements, size_t size, void** result, cudaStream_t stream)
{
	return CompressLightweight(elements, size, (unsigned char**)result, stream);
}

size_t Compression::DecompressTrunk(void* data, size_t size, storeElement** result)
{
	return DecompressLightweight(static_cast<unsigned char*>(data), size, result);;
}

} /* namespace task */
} /* namespace ddj */
