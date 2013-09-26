/*
 * StoreIncludes.h
 *
 *  Created on: Aug 10, 2013
 *      Author: Karol Dzitkowski
 *
 *  This file includes all required headers from outside libs like pantheios or boost
 *  It also contains global inline helper functions and common used typedefs. Helper
 *  macros also should be stored here.
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
#include <stdexcept>

/* CUDA */
#include <cuda.h>
#include <cuda_runtime_api.h>

/* BOOST */
#include <boost/array.hpp>
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

/* PANTHEIOS */
#include <pantheios/pantheios.hpp>
#include <pantheios/frontends/stock.h>
#include <pantheios/inserters/args.hpp>
#include <pantheios/backend.h>
#include <pantheios/inserters/threadid.hpp>
#include <pantheios/inserters/integer.hpp>

/* OTHER */
#include "../BTree/btree.h"

/* DEFINES */
#define STORE_BUFFER_SIZE 2
#define PSTR(x)         PANTHEIOS_LITERAL_STRING(x)
#ifndef ERROR_WHEN_FALSE
#define ERROR_WHEN_FALSE(expression) \
({ \
    if(false == (expression) ){ \
    fprintf(stderr,"%s:%d\n",__FILE__,__LINE__); \
    kill(0,SIGKILL); \
    exit(EXIT_FAILURE);} \
})
#endif

/* TYPEDEFS */
typedef unsigned long long int ullint;
typedef stx::btree<ullint, int> tree;
typedef tree* tree_pointer;
typedef int tag_type;
typedef float store_value_type;
typedef float info_value_type;

#endif /* DDJ_STOREINCLUDES_H_ */
