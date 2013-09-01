/*
 * GpuUploaderMonitor.cpp
 *
 *  Created on: Aug 31, 2013
 *      Author: parallels
 */

#include "GpuUploaderMonitor.h"

namespace ddj {
namespace store {

GpuUploaderMonitor::GpuUploaderMonitor(BTreeMonitor* bTreeInserter) {
	this->_bTreeInserter = bTreeInserter;
}

GpuUploaderMonitor::~GpuUploaderMonitor() {
}

} /* namespace store */
} /* namespace ddj */
