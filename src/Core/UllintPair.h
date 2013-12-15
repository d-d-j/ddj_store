/*
 * UllintPair.h
 *
 *  Created on: 15-12-2013
 *      Author: ghash
 */

#ifndef ULLINTPAIR_H_
#define ULLINTPAIR_H_

#include "../Store/StoreTypedefs.h"

namespace ddj {

	struct ullintPair
	{
		ullint first;
		ullint second;

		ullintPair():first(1),second(1){}
		ullintPair(ullint f, ullint s):first(f),second(s){}
		ullintPair(const ullintPair& cp)
		{
			first = cp.first;
			second = cp.second;
		}
		~ullintPair(){}

		bool operator< (const ullintPair& rhs) const
		{
			if(first < rhs.first && second < rhs.second) return true;
			else return false;
		}

		bool operator== (const ullintPair& rhs) const
		{
			if(first >= rhs.first && second <= rhs.second) return true;
			if(first <= rhs.first && second >= rhs.second) return true;
			return false;
		}

		bool isIntersecting(ullintPair rhs)
		{
			if(first > rhs.first && first < rhs.second) return true;
			if(second > rhs.first && second < rhs.second) return true;
			if(rhs.first > first && rhs.first < second) return true;
			if(rhs.second > first && rhs.second < second) return true;
			return false;
		}
	};

} /* namespace ddj */
#endif /* ULLINTPAIR_H_ */
