#include "QueryTest.h"
namespace ddj {
namespace query {

	TEST_F(QueryTest, Constructor)
	{
		char input[] = {1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		auto actual = Query((void*)input).toString();
		auto expected = "query[aggregationType: 0; metrics: 1; tags: 1 0; timePeriods: (11,21) (5,7)]";
		EXPECT_EQ(expected, actual);
	}

}
}
