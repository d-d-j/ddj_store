/*
 * Semaphore.h
 *
 *  Created on: 30-10-2013
 *      Author: ghash
 */

#ifndef SEMAPHORE_H_
#define SEMAPHORE_H_

#include <boost/thread.hpp>

namespace ddj {

	class Semaphore
	{
		unsigned int _max;
		unsigned int _value;
		boost::mutex _mutex;
		boost::condition_variable _cond;
	public:
		Semaphore(unsigned int max);
		virtual ~Semaphore();

		void Wait();
		void Release();
	};

} /* namespace ddj */
#endif /* SEMAPHORE_H_ */
