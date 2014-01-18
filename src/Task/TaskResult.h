#ifndef TASKRESULT_H_
#define TASKRESULT_H_

#include "TaskType.h"
#include <string>
#include <sstream>
#include <cstring>

namespace ddj {
namespace task {

	struct taskResult
	{
	public:
		int64_t task_id;
		TaskType type;
		int32_t result_size;
		void* result_data;

		taskResult(
				int64_t taskId,
				TaskType type,
				void* resultData = nullptr,
				int32_t resultSize = 0,
				char* message = nullptr):
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

		~taskResult(){}

		bool operator== (const taskResult& rhs) const
		{
			if(task_id == rhs.task_id && type == rhs.type &&
					result_size == rhs.result_size)
			{
				return true;
			}
			return false;
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
