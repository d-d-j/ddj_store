/*
 * BTreeMonitorTest.h
 *
 *  Created on: 13-12-2013
 *      Author: ghash
 */

#ifndef BTREEMONITORTEST_H_
#define BTREEMONITORTEST_H_

#include "../BTree/BTreeMonitor.h"
#include <gtest/gtest.h>

namespace ddj {
namespace unittest {

	class BTreeMonitorTest : public testing::Test {
	protected:
		virtual void SetUp() {

		  }

		btree::BTreeMonitor* _monitor;
	};

	TEST_F(BTreeMonitorTest, Constructor)
	{
		_monitor = new btree::BTreeMonitor(1);
		EXPECT_TRUE(_monitor != NULL);
	}



} /* namespace unittest */
} /* namespace ddj */
#endif /* BTREEMONITORTEST_H_ */
