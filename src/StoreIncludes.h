/*
 * StoreIncludes.h
 *
 *  Created on: Aug 10, 2013
 *      Author: Karol Dzitkowski
 *
 *  This file includes all required headers from outside libs like pantheios or boost
 *  It also contains macros and additional typedefs.
 */

#ifndef DDJ_STOREINCLUDES_H_
#define DDJ_STOREINCLUDES_H_

/* COMMON */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <iostream>
#include <algorithm>
#include <memory>
#include <cstddef>
#include <stdexcept>

/* BOOST */
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/container/vector.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <boost/signals2/signal.hpp>
#include <boost/container/container_fwd.hpp>

#endif /* DDJ_STOREINCLUDES_H_ */
