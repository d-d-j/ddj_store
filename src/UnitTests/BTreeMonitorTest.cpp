#include "BTreeMonitorTest.h"

namespace ddj {
namespace btree {

	TEST_F(BTreeMonitorTest, Constructor)
	{
		// CHECK
		EXPECT_TRUE(_monitor != nullptr);
		auto result = _monitor->SelectAll();
		ASSERT_TRUE(result->empty());
	}

	TEST_F(BTreeMonitorTest, Insert_One)
	{
		// PREPARE
		store::storeTrunkInfo* elem = new store::storeTrunkInfo(1, 2, 3, 4, 5);
		_monitor->Insert(elem);

		// TEST
		auto result = _monitor->SelectAll();

		// CHECK
		ASSERT_FALSE(result->empty());
		EXPECT_EQ(1, result->size());

		// CLEAN
		delete result;
		delete elem;
	}

	TEST_F(BTreeMonitorTest, Insert_Many)
	{
		// PREPARE
		store::storeTrunkInfo* elem0 = new store::storeTrunkInfo(1, 2, 3, 4, 5);
		store::storeTrunkInfo* elem1 = new store::storeTrunkInfo(6, 7, 8, 9, 10);
		store::storeTrunkInfo* elem2 = new store::storeTrunkInfo(11, 12, 13, 14, 15);
		store::storeTrunkInfo* elem3 = new store::storeTrunkInfo(16, 17, 18, 19, 20);
		_monitor->Insert(elem0);
		_monitor->Insert(elem1);
		_monitor->Insert(elem2);
		_monitor->Insert(elem3);

		// TEST
		auto result = _monitor->SelectAll();

		// CHECK
		ASSERT_FALSE(result->empty());
		EXPECT_EQ(4, result->size());

		// CLEAN
		delete result;
		delete elem0;
		delete elem1;
		delete elem2;
		delete elem3;
	}

	/*
	trunks		|------||------||------|
	*/
	TEST_F(BTreeMonitorTest, SelectAll_SeparateTrunks)
	{
		// PREPARE
		store::storeTrunkInfo* elem0 = new store::storeTrunkInfo(1, 2, 3, 4, 8);
		store::storeTrunkInfo* elem1 = new store::storeTrunkInfo(1, 4, 5, 9, 13);
		store::storeTrunkInfo* elem2 = new store::storeTrunkInfo(1, 6, 7, 14, 18);

		/* elements in B+Tree will be sorted to: elem0, elem1, elem2 */
		_monitor->Insert(elem0);
		_monitor->Insert(elem2);
		_monitor->Insert(elem1);

		// TEST
		auto result = _monitor->SelectAll();

		// CHECK
		EXPECT_EQ(4, result->data()[0].first);
		EXPECT_EQ(8, result->data()[0].second);
		EXPECT_EQ(9, result->data()[1].first);
		EXPECT_EQ(13, result->data()[1].second);
		EXPECT_EQ(14, result->data()[2].first);
		EXPECT_EQ(18, result->data()[2].second);

		// CLEAN
		delete result;
		delete elem0;
		delete elem1;
		delete elem2;
	}

	/*
	trunks			|------|
				|------||------|
	*/
	TEST_F(BTreeMonitorTest, SelectAll_IntersectingTrunks)
	{
		// PREPARE
		store::storeTrunkInfo* elem0 = new store::storeTrunkInfo(1, 2, 8, 0, 7);
		store::storeTrunkInfo* elem1 = new store::storeTrunkInfo(1, 6, 18, 8, 15);
		store::storeTrunkInfo* elem2 = new store::storeTrunkInfo(1, 12, 22, 16, 23);

		/* elements in B+Tree will be sorted to: elem0, elem1, elem2 */
		_monitor->Insert(elem0);
		_monitor->Insert(elem2);
		_monitor->Insert(elem1);

		// TEST
		auto result = _monitor->SelectAll();

		// CHECK
		EXPECT_EQ(0, result->data()[0].first);
		EXPECT_EQ(7, result->data()[0].second);
		EXPECT_EQ(8, result->data()[1].first);
		EXPECT_EQ(15, result->data()[1].second);
		EXPECT_EQ(16, result->data()[2].first);
		EXPECT_EQ(23, result->data()[2].second);

		// CLEAN
		delete result;
		delete elem0;
		delete elem1;
		delete elem2;
	}

	/*
	trunks		|------|------|
	*/
	TEST_F(BTreeMonitorTest, SelectAll_OneEndAndAnotherBeginEqual)
	{
		// PREPARE
		store::storeTrunkInfo* elem0 = new store::storeTrunkInfo(1, 2, 8, 0, 7);
		store::storeTrunkInfo* elem1 = new store::storeTrunkInfo(1, 8, 18, 8, 15);
		_monitor->Insert(elem0);
		_monitor->Insert(elem1);

		// TEST
		auto result = _monitor->SelectAll();

		// CHECK
		EXPECT_EQ(0, result->data()[0].first);
		EXPECT_EQ(7, result->data()[0].second);
		EXPECT_EQ(8, result->data()[1].first);
		EXPECT_EQ(15, result->data()[1].second);

		// CLEAN
		delete result;
		delete elem0;
		delete elem1;
	}

	/*
	period		<-------------->
	trunks			|------|
	*/
	TEST_F(BTreeMonitorTest, Select_OneOne_TrunkInTimePeriod)
	{
		// PREAPRE
		store::storeTrunkInfo* elem = new store::storeTrunkInfo(1, 2, 8, 0, 7);
		_monitor->Insert(elem);
		ullintPair period{0,10};
		boost::container::vector<ullintPair> timePeriods;
		timePeriods.push_back(period);

		// TEST
		auto result = _monitor->Select(timePeriods);

		// CHECK
		ASSERT_FALSE(result->empty());
		EXPECT_EQ(1, result->size());
		EXPECT_EQ(0, result->data()[0].first);
		EXPECT_EQ(7, result->data()[0].second);

		// CLEAN
		delete result;
		delete elem;
	}

	/*
	period				<---->
	trunks			|-----------|
	*/
	TEST_F(BTreeMonitorTest, Select_OneOne_TimePeriodInTrunk)
	{
		// PREAPRE
		store::storeTrunkInfo* elem = new store::storeTrunkInfo(1, 0, 10, 0, 7);
		ullintPair period{2,8};
		boost::container::vector<ullintPair> timePeriods;
		timePeriods.push_back(period);
		_monitor->Insert(elem);

		// TEST
		auto result = _monitor->Select(timePeriods);

		// CHECK
		ASSERT_FALSE(result->empty());
		EXPECT_EQ(1, result->size());
		EXPECT_EQ(0, result->data()[0].first);
		EXPECT_EQ(7, result->data()[0].second);

		// CLEAN
		delete result;
		delete elem;
	}

	/*
	period		<-------->
	trunks			|---------|
	*/
	TEST_F(BTreeMonitorTest, Select_OneOne_TimePeriodEndInTrunk)
	{
		// PREAPRE
		store::storeTrunkInfo* elem = new store::storeTrunkInfo(1, 4, 10, 0, 7);
		ullintPair period{0,5};
		boost::container::vector<ullintPair> timePeriods;
		timePeriods.push_back(period);
		_monitor->Insert(elem);

		// TEST
		auto result = _monitor->Select(timePeriods);

		// CHECK
		ASSERT_FALSE(result->empty());
		EXPECT_EQ(1, result->size());
		EXPECT_EQ(0, result->data()[0].first);
		EXPECT_EQ(7, result->data()[0].second);

		// CLEAN
		delete result;
		delete elem;
	}

	/*
	period				<-------->
	trunks			|---------|
	*/
	TEST_F(BTreeMonitorTest, Select_OneOne_TimePeriodBeginInTrunk)
	{
		// PREAPRE
		store::storeTrunkInfo* elem = new store::storeTrunkInfo(1, 4, 10, 0, 7);
		ullintPair period{5,15};
		boost::container::vector<ullintPair> timePeriods;
		timePeriods.push_back(period);
		_monitor->Insert(elem);

		// TEST
		auto result = _monitor->Select(timePeriods);

		// CHECK
		ASSERT_FALSE(result->empty());
		EXPECT_EQ(1, result->size());
		EXPECT_EQ(0, result->data()[0].first);
		EXPECT_EQ(7, result->data()[0].second);

		// CLEAN
		delete result;
		delete elem;
	}

	/*
	period				<-------->
	trunks	|---------|
	*/
	TEST_F(BTreeMonitorTest, Select_OneOne_TimePeriodOutsideTrunk_Greater)
	{
		// PREAPRE
		store::storeTrunkInfo* elem = new store::storeTrunkInfo(1, 4, 10, 0, 7);
		ullintPair period{11,15};
		boost::container::vector<ullintPair> timePeriods;
		timePeriods.push_back(period);
		_monitor->Insert(elem);

		// TEST
		auto result = _monitor->Select(timePeriods);

		// CHECK
		ASSERT_TRUE(result->empty());

		// CLEAN
		delete result;
		delete elem;
	}

	/*
	period	<-------->
	trunks				|---------|
	*/
	TEST_F(BTreeMonitorTest, Select_OneOne_TimePeriodOutsideTrunk_Lower)
	{
		// PREAPRE
		store::storeTrunkInfo* elem = new store::storeTrunkInfo(1, 9, 19, 0, 7);
		ullintPair period{1,8};
		boost::container::vector<ullintPair> timePeriods;
		timePeriods.push_back(period);
		_monitor->Insert(elem);

		// TEST
		auto result = _monitor->Select(timePeriods);

		// CHECK
		ASSERT_TRUE(result->empty());

		// CLEAN
		delete result;
		delete elem;
	}

	/*
	period				<-------->
	trunks		|---||---||---||---||---|
	*/
	TEST_F(BTreeMonitorTest, Select_ManyOne_SeparateTrunks)
		{
			// PREAPRE
			store::storeTrunkInfo* elem0 = new store::storeTrunkInfo(1, 0, 9, 0, 9);
			store::storeTrunkInfo* elem1 = new store::storeTrunkInfo(1, 10, 19, 10, 19);
			store::storeTrunkInfo* elem2 = new store::storeTrunkInfo(1, 20, 29, 20, 29);
			store::storeTrunkInfo* elem3 = new store::storeTrunkInfo(1, 30, 39, 30, 39);
			store::storeTrunkInfo* elem4 = new store::storeTrunkInfo(1, 40, 49, 40, 49);
			_monitor->Insert(elem0);
			_monitor->Insert(elem1);
			_monitor->Insert(elem2);
			_monitor->Insert(elem3);
			_monitor->Insert(elem4);
			ullintPair period{11,38};
			boost::container::vector<ullintPair> timePeriods;
			timePeriods.push_back(period);

			// TEST
			auto result = _monitor->Select(timePeriods);

			// CHECK
			ASSERT_FALSE(result->empty());
			EXPECT_EQ(3, result->size());
			EXPECT_EQ(10, result->data()[0].first);
			EXPECT_EQ(19, result->data()[0].second);
			EXPECT_EQ(20, result->data()[1].first);
			EXPECT_EQ(29, result->data()[1].second);
			EXPECT_EQ(30, result->data()[2].first);
			EXPECT_EQ(39, result->data()[2].second);

			// CLEAN
			delete result;
			delete elem0;
			delete elem1;
			delete elem2;
			delete elem3;
			delete elem4;
		}

	/*
	period	 <--------->		 <--------->
	trunks		|---||---||---||---||---|
	*/
	TEST_F(BTreeMonitorTest, Select_ManyMany_SeparateTrunks)
	{
		// PREAPRE
		store::storeTrunkInfo* elem0 = new store::storeTrunkInfo(1, 10, 19, 10, 19);
		store::storeTrunkInfo* elem1 = new store::storeTrunkInfo(1, 20, 29, 20, 29);
		store::storeTrunkInfo* elem2 = new store::storeTrunkInfo(1, 30, 39, 30, 39);
		store::storeTrunkInfo* elem3 = new store::storeTrunkInfo(1, 40, 49, 40, 49);
		store::storeTrunkInfo* elem4 = new store::storeTrunkInfo(1, 50, 59, 50, 59);
		_monitor->Insert(elem0);
		_monitor->Insert(elem1);
		_monitor->Insert(elem2);
		_monitor->Insert(elem3);
		_monitor->Insert(elem4);
		ullintPair period0{0,25};
		ullintPair period1{45,70};
		boost::container::vector<ullintPair> timePeriods;
		timePeriods.push_back(period0);
		timePeriods.push_back(period1);

		// TEST
		auto result = _monitor->Select(timePeriods);

		// CHECK
		ASSERT_FALSE(result->empty());
		EXPECT_EQ(4, result->size());
		EXPECT_EQ(10, result->data()[0].first);
		EXPECT_EQ(19, result->data()[0].second);
		EXPECT_EQ(20, result->data()[1].first);
		EXPECT_EQ(29, result->data()[1].second);
		EXPECT_EQ(40, result->data()[2].first);
		EXPECT_EQ(49, result->data()[2].second);
		EXPECT_EQ(50, result->data()[3].first);
		EXPECT_EQ(59, result->data()[3].second);

		// CLEAN
		delete result;
		delete elem0;
		delete elem1;
		delete elem2;
		delete elem3;
		delete elem4;
	}

	/*
	period			  <-------->
	trunks	|---------|
	*/
	TEST_F(BTreeMonitorTest, Select_OneOne_TimePeriodLeftOneElemIntersect)
	{
		// PREAPRE
		store::storeTrunkInfo* elem = new store::storeTrunkInfo(1, 4, 10, 0, 7);
		ullintPair period{10,15};
		boost::container::vector<ullintPair> timePeriods;
		timePeriods.push_back(period);
		_monitor->Insert(elem);

		// TEST
		auto result = _monitor->Select(timePeriods);

		// CHECK
		ASSERT_EQ(1, result->size());
		EXPECT_EQ(0, (*result)[0].first);
		EXPECT_EQ(7, (*result)[0].second);

		// CLEAN
		delete result;
		delete elem;
	}


} /* namespace btree */
} /* namespace ddj */
