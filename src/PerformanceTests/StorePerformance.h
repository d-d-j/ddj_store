/*
 * StorePerformance.h
 *
 *  Created on: 11-01-2014
 *      Author: ghash
 */

#ifndef STOREPERFORMANCE_H_
#define STOREPERFORMANCE_H_

#include "../Core/Logger.h"
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <fstream>

namespace ddj {
namespace task {

	class StorePerformance : public ::testing::TestWithParam<int>
	{
	protected:
		StorePerformance();
		virtual ~StorePerformance();

		virtual void SetUp()
		{

		}
	};

} /* namespace task */
} /* namespace ddj */
#endif /* STOREPERFORMANCE_H_ */
