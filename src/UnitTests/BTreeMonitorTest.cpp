#include "BTreeMonitorTest.h"

namespace ddj {
namespace unittest {

	TEST_F(BTreeMonitorTest, Constructor)
	{
		EXPECT_TRUE(_monitor != NULL);
	}

	TEST_F(BTreeMonitorTest, InsertOne)
	{
		store::storeTrunkInfo* elem = new store::storeTrunkInfo(1, 2, 3, 4, 5);
		_monitor->Insert(elem);
		auto result = _monitor->SelectAll();

		ASSERT_FALSE(result->empty());
		EXPECT_EQ(1, result->size());
		EXPECT_EQ(4, result->data()[0].start);
		EXPECT_EQ(5, result->data()[0].end);

		delete result;
		delete elem;
	}

} /* namespace unittest */
} /* namespace ddj */
