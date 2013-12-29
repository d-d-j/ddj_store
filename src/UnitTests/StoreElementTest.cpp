#include "StoreElementTest.h"

#include <algorithm>
#include <boost/foreach.hpp>

namespace ddj {
namespace store {

	TEST_F(StoreElementTest, Constructor)
	{
		EXPECT_TRUE(_element != nullptr);
		ASSERT_EQ(24, sizeof(*_element) );
	}

	TEST_F(StoreElementTest, Sort)
	{
		const int len = 10;
		int mod = 3;
		storeElement elements[len];

		for (int i=0;i<len;i++)
	    {
	        elements[i].time = len-i-1;
	    }

		std::sort(elements, elements+len, OrderStoreElementByTimeAsc());

		for (int i=0;i<len;i++)
	    {
	        ASSERT_EQ(elements[i].time, i);
	    }
	}
}}
