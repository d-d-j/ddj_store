#include "TaskMonitorTest.h"
#include <vector>

namespace ddj {
namespace task {

	TEST_F(TaskMonitorTest, Constructor)
	{
	    EXPECT_TRUE(taskMonitor != nullptr);
	}

	TEST_F(TaskMonitorTest, PopCompletedTasks_Should_Removed_Only_Tasks_That_Are_Returned)
	{
		//simulate normal work adding and completing task
		for (int iteration=1;iteration<4;iteration++) {
		    int tasksCount = 6;
		    //Add some tasks
		    for (int i=0;i<tasksCount;i++) {
		    	taskMonitor->AddTask(iteration*10+i, TaskType::Insert, nullptr, 0);
			}

			//Mark half of them as complete
			std::vector<int64_t> completedTaskId;
			std::vector<int64_t> notCompletedTaskId;
			for (unsigned int i=0;i<taskMonitor->_tasks.size();i++) {
				auto id = taskMonitor->_tasks[i]->_taskId;
		    	taskMonitor->_tasks[i]->_isCompleted = id % iteration;
		    	if (taskMonitor->_tasks[i]->_isCompleted == true)
		    		completedTaskId.push_back(id);
		    	else
		    		notCompletedTaskId.push_back(id);
			}

			auto actual = taskMonitor->PopCompletedTasks();

			ASSERT_EQ(completedTaskId.size(), actual.size());
			for (unsigned int i=0;i<completedTaskId.size();i++) {
		    	EXPECT_EQ(completedTaskId[i], actual[i]->_taskId);
			}
			ASSERT_TRUE(completedTaskId.size() <= actual.size());
			for (unsigned int i=0;i<notCompletedTaskId.size();i++) {
		    	EXPECT_EQ(notCompletedTaskId[i], taskMonitor->_tasks[i]->_taskId);
			}
		}

	}

}}