#include "TaskTest.h"

namespace ddj {
namespace task {

	TEST_F(TaskTest, Constructor)
	{
		EXPECT_TRUE(_task != NULL);
	}

	TEST_F(TaskTest, AppendMessage)
	{
		const char *foo = "Foo";
		const char *bar = " bar";
		const char *expected = "Foo bar";

		_task->appendMessage(foo);
		_task->appendMessage(NULL);
		_task->appendMessage(bar);

		EXPECT_EQ(std::string(expected), std::string(_task->_message));
	}

}}
