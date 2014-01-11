/*
 * QueryCorePerformance.h
 *
 *  Created on: 11-01-2014
 *      Author: ghash
 */

#ifndef QUERYCOREPERFORMANCE_H_
#define QUERYCOREPERFORMANCE_H_

#include "../Core/Logger.h"
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <fstream>

namespace ddj {
namespace task {

	class QueryCorePerformance : public ::testing::TestWithParam<int>
	{
	protected:
		QueryCorePerformance();
		virtual ~QueryCorePerformance();

		virtual void SetUp()
		{

		}
	};

} /* namespace task */
} /* namespace ddj */
#endif /* QUERYCOREPERFORMANCE_H_ */
