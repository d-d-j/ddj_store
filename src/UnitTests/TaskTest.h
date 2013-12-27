#ifndef TASKTEST_H
#define TASKTEST_H

#include "../Task/Task.h"
#include <gtest/gtest.h>

namespace ddj {
namespace task {

	class TaskTest : public testing::Test {
	protected:
		virtual void SetUp()
		{
			_task = new Task(
								0,
								TaskType::Error,
								NULL,
								0,
								NULL);
		}

		virtual void TearDown() {
    		delete _task;
		}

		Task* _task;
	};

}}
#endif // TASKTEST_H
