#include "QueryTest.h"
namespace ddj {
namespace query {

	TEST_F(QueryTest, Constructor)
	{
		unsigned char input[] = {
			1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0,
			7, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		};
		auto actual = Query((void*)input).toString();
		auto expected = "query[aggregationType: 0; metrics: 1; tags: 1 0; timePeriods: (11,21) (5,7)]";
		EXPECT_EQ(expected, actual);
	}

	TEST_F(QueryTest, Constructor_With_Additional_Data)
	{
		unsigned char input[] = {
			1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0,
			7, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0,
			10, 0, 0, 0, 5, 0, 0, 0, 0,	0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0
		};
		auto actual = Query((void*)input).toString();
		auto expected = "query[aggregationType: 10; metrics: 1; tags: 1 0; timePeriods: (11,21) (5,7)]";
		EXPECT_EQ(expected, actual);

		auto data = (data::histogramTimeData*)Query((void*)input).aggregationData;
		ASSERT_FALSE(data == nullptr);
		EXPECT_EQ( 5, data->min);
		EXPECT_EQ(10, data->max);
		EXPECT_EQ(15, data->bucketCount);


	}

		TEST_F(QueryTest, Constructor_With_Additional_Data_For_Value_Histogram)
	{
		unsigned char input[] = {
			1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0,
			7, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0,
			9, 0, 0, 0, 0, 0, 160, 64, 0, 0, 32, 65, 15, 0, 0, 0
		};
		auto actual = Query((void*)input).toString();
		auto expected = "query[aggregationType: 9; metrics: 1; tags: 1 0; timePeriods: (11,21) (5,7)]";
		EXPECT_EQ(expected, actual);

		auto data = (data::histogramValueData*)Query((void*)input).aggregationData;
		ASSERT_FALSE(data == nullptr);
		EXPECT_EQ( 5, data->min);
		EXPECT_EQ(10, data->max);
		EXPECT_EQ(15, data->bucketCount);


	}
}
}
