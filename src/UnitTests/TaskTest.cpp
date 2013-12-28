#include "TaskTest.h"
#include <iostream>

namespace ddj {
namespace task {

	TEST_F(TaskTest, Constructor)
	{
		ASSERT_TRUE(_task != nullptr);
		EXPECT_FALSE(_task->IsCompleated());
		EXPECT_EQ(nullptr, _task->GetResult());
	}

	TEST_F(TaskTest, AppendMessage)
	{
		// PREPARE DATA
		const char *foo = "Foo";
		const char *bar = " bar";
		const char *expected = "Foo bar";

		// TEST
		_task->appendMessage(foo);
		_task->appendMessage(nullptr);
		_task->appendMessage(bar);

		// CHECK
		EXPECT_EQ(std::string(expected), std::string(_task->_message));
	}

	TEST_F(TaskTest, SetResult_SingleResult)
	{
		// PREPARE DATA
		boost::condition_variable condVar;
		Task task(1, TaskType::Insert, nullptr, 1, &condVar);

		// TEST
		task.SetResult(true, "success", nullptr, 0);

		// CHECK
		ASSERT_TRUE(task.IsCompleated());
		EXPECT_EQ(nullptr, task.GetResult()->result_data);
		EXPECT_EQ(0, task.GetResult()->result_size);
		EXPECT_EQ(1, task.GetResult()->task_id);
		EXPECT_EQ(TaskType::Insert, task.GetResult()->type);
		//EXPECT_STREQ("success", task.GetResult()->message);
	}

	TEST_F(TaskTest, SetResult_MultipleResults_Info_Success)
	{
		// PREPARE DATA
		boost::condition_variable condVar;
		Task task(1, TaskType::Info, nullptr, 3, &condVar);
		store::storeNodeInfo* info_1 = new store::storeNodeInfo(199,198,177,1);
		store::storeNodeInfo* info_2 = new store::storeNodeInfo(299,28,277,266);
		store::storeNodeInfo* info_3 = new store::storeNodeInfo(399,0,17,6);
		int infoSize = sizeof(store::storeNodeInfo);

		// PREPARE EXPECTED RESULT
		taskResult expected_result(1, TaskType::Info, nullptr, 3*infoSize, (char*)"success_1;success_2;success_3;");

		// TEST
		task.SetResult(true, "success_1;", info_1, infoSize);

		// CHECK
		ASSERT_FALSE(task.IsCompleated());
		EXPECT_EQ(nullptr, task.GetResult());

		// TEST
		task.SetResult(true, "success_2;", info_2, infoSize);

		// CHECK
		ASSERT_FALSE(task.IsCompleated());
		EXPECT_EQ(nullptr, task.GetResult());

		// TEST
		task.SetResult(true, "success_3;", info_3, infoSize);

		// CHECK
		ASSERT_TRUE(task.IsCompleated());
		EXPECT_EQ(expected_result, *task.GetResult());
		EXPECT_EQ(*info_1, ((store::storeNodeInfo*)task.GetResult()->result_data)[0]);
		EXPECT_EQ(*info_2, ((store::storeNodeInfo*)task.GetResult()->result_data)[1]);
		EXPECT_EQ(*info_3, ((store::storeNodeInfo*)task.GetResult()->result_data)[2]);
	}

} /* namespace task */
} /* namespace ddj */
