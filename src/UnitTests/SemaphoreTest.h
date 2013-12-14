#ifndef SEMAPHORETEST_H_
#define SEMAPHORETEST_H_

#include "../Core/Semaphore.h"
#include <gtest/gtest.h>

namespace ddj {
namespace unittest {

	class SemaphoreTest : public testing::Test {
	protected:
		virtual void SetUp()
		{
			_semaphore = new Semaphore(1);
		}

		virtual void TearDown() {
    		delete _semaphore;
		}

		Semaphore* _semaphore;
	};

}
}
#endif