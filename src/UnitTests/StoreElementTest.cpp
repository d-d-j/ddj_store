#include "StoreElementTest.h"

namespace ddj {
namespace store {

	TEST_F(StoreElementTest, Constructor)
	{
		EXPECT_TRUE(_element != NULL);
		ASSERT_EQ(24, sizeof(*_element) );
	}
}}
