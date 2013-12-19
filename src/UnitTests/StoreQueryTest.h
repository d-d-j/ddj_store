#ifndef STOREQUERYTEST_H
#define STOREQUERYTEST_H


#include <gtest/gtest.h>
#include "../Store/StoreQuery.h"
namespace ddj {
namespace store {

	class StoreQueryTest : public testing::Test {

	protected:
		virtual void SetUp()
		{
		}

		virtual void TearDown() {

    		delete _storeQuery;
		}
	storeQuery* _storeQuery;

	};
}
}
#endif // STOREQUERYTEST_H
