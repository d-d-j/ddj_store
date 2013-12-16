/*
 * TaskType.h
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#ifndef TASKTYPE_H_
#define TASKTYPE_H_

namespace ddj {
namespace task {

	enum TaskType
	{
		Error = 0,
		Insert = 1,
		Select = 2,
		Flush = 3,
		Info = 4
	};

} /* namespace task */
} /* namespace ddj */

#endif /* TASKTYPE_H_ */
