/*
 * Semaphore.cpp
 *
 *  Created on: 30-10-2013
 *      Author: ghash
 */

#include "Semaphore.h"

namespace ddj {

	Semaphore::Semaphore(unsigned int max)
	{
		_max = max;
		_value = 0;
	}

	Semaphore::~Semaphore(){}

	int Semaphore::Wait()
	{
		boost::mutex::scoped_lock lock(_mutex);
		if(_value >= _max)
			_cond.wait(lock);
		_value++;
		return _value;
	}

	void Semaphore::Release()
	{
		boost::mutex::scoped_lock lock(_mutex);
		_value--;
		_cond.notify_one();
	}

} /* namespace ddj */
