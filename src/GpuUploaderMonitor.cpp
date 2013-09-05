/*
 * GpuUploaderMonitor.cpp
 *
 *  Created on: Aug 31, 2013
 *      Author: parallels
 */

#include "GpuUploaderMonitor.h"

namespace ddj {
namespace store {

GpuUploaderMonitor::GpuUploaderMonitor(BTreeMonitor* bTreeInserter)
{
	this->_bTreeInserter = bTreeInserter;

}

GpuUploaderMonitor::~GpuUploaderMonitor()
{

}

bool GpuUploaderMonitor::SendStoreElementsToGpu(boost::array<storeElement, STORE_BUFFER_SIZE>* elements)
{
	this->_elementsToUpload = elements->c_array();
	this->_readyToUpload = 0;
	return false;
}

} /* namespace store */
} /* namespace ddj */
