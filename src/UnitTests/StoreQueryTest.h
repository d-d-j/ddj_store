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
		char input[] = {1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
			_storeQuery = new storeQuery((void*)input);
		}

		virtual void TearDown() {

    		delete _storeQuery;
		}
	storeQuery* _storeQuery;

	};
}
}
#endif // STOREQUERYTEST_H
