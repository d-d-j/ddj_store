/*
 * BTreeMonitorTest.h
 *
 *  Created on: 13-12-2013
 *      Author: ghash
 */

#ifndef BTREEMONITORTEST_H_
#define BTREEMONITORTEST_H_

#include "../BTree/BTreeMonitor.h"
 #include "../Core/Logger.h"
#include <gtest/gtest.h>

namespace ddj {
namespace btree {

	class BTreeMonitorTest : public testing::Test
	{
	protected:
		virtual void SetUp()
		{
			_monitor = new btree::BTreeMonitor(1);
		}

		virtual void TearDown()
		{
			delete _monitor;
		}

		btree::BTreeMonitor* _monitor;
	};

} /* namespace btree */
} /* namespace ddj */
#endif /* BTREEMONITORTEST_H_ */
