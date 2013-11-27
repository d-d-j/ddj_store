#ifndef TASKRESULT_H_
#define TASKRESULT_H_

#include "TaskType.h"
#include <string>
#include <sstream>

namespace ddj {
namespace task {

	struct taskResult
	{
	public:
		int task_id;
		TaskType type;
		int result_size;
		void* result_data;

		taskResult(
				int taskId,
				TaskType type,
				void* resultData = nullptr,
				int resultSize = 0):
					task_id(taskId),
					type(type),
					result_size(resultSize),
					result_data(resultData){}

		taskResult(const taskResult & result)
		{
			task_id = result.task_id;
			type = result.type;
			result_size = result.result_size;
			result_data = result.result_data;
		}

		std::string toString()
		{
			 std::ostringstream stream;
		     stream << "["<<task_id<<", "<<type<<", "<<result_size<<"]";
		     return  stream.str();
		}
	};

} /* namespace task */
} /* namespace ddj */

#endif /* TASKRESULT_H_ */
