#ifndef TASKMONITORTEST_H_
#define TASKMONITORTEST_H_

#include "../Task/TaskMonitor.h"
#include "../Task/Task.h"
#include "../Task/TaskType.h"
#include <gtest/gtest.h>

namespace ddj {
namespace task {

	class TaskMonitorTest : public testing::Test {
	protected:
		virtual void SetUp()
		{
			taskMonitor = new TaskMonitor(nullptr);
		}

		virtual void TearDown() {
    		delete taskMonitor;
		}

		TaskMonitor* taskMonitor;
	};

}}
#endif // TASKMONITORTEST_H_
