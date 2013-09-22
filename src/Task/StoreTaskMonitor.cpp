/*
 * StoreTaskMonitor.cpp
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#include "StoreTaskMonitor.h"

namespace ddj
{
namespace store
{

StoreTaskMonitor::StoreTaskMonitor(boost::condition_variable* condResponseReady)
{
	this->_condResponseReady = condResponseReady;
}

StoreTaskMonitor::~StoreTaskMonitor()
{
	// TODO Auto-generated destructor stub
}

} /* namespace store */
} /* namespace ddj */
