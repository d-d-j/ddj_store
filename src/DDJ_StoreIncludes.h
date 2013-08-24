/*
 * DDJ_StoreIncludes.h
 *
 *  Created on: Aug 10, 2013
 *      Author: parallels
 */

#ifndef DDJ_STOREINCLUDES_H_
#define DDJ_STOREINCLUDES_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <iostream>
#include <boost/array.hpp>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <c++/4.6/ext/hash_map>
#include <pantheios/pantheios.hpp>
#include <pantheios/frontends/stock.h>
#include <pantheios/inserters/args.hpp>
#include <pantheios/backend.h>
#include <pantheios/inserters/threadid.hpp>
#include <pantheios/inserters/integer.hpp>
#include "btree.h"

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

inline void h_LogThreadInfo(const char* text)
{
	try
	{
		pantheios::log_INFORMATIONAL(PSTR("[Thread id="), boost::lexical_cast<std::string>(boost::this_thread::get_id()), PSTR("] "), PSTR(text));
	}
	catch(std::bad_alloc&)
	{
		pantheios::log(pantheios::alert, PSTR("out of memory"));
	}
	catch(std::exception& x)
	{
		pantheios::log_CRITICAL(PSTR("Exception: "), x);
	}
	catch(...)
	{
		pantheios::logputs(pantheios::emergency, PSTR("Unexpected unknown error"));
		}
}

inline void h_LogThreadDebug(const char* text)
{
	try
	{
		pantheios::log_DEBUG(PSTR("[Thread id="), boost::lexical_cast<std::string>(boost::this_thread::get_id()), PSTR("] "), PSTR(text));
	}
	catch(std::bad_alloc&)
	{
		pantheios::log(pantheios::alert, PSTR("out of memory"));
	}
	catch(std::exception& x)
	{
		pantheios::log_CRITICAL(PSTR("Exception: "), x);
	}
	catch(...)
	{
		pantheios::logputs(pantheios::emergency, PSTR("Unexpected unknown error"));
	}
}

inline void h_LogThreadError(const char* text)
{
	try
	{
		pantheios::log_ERROR(PSTR("[Thread id="), boost::lexical_cast<std::string>(boost::this_thread::get_id()), PSTR("] "), PSTR(text));
	}
	catch(std::bad_alloc&)
	{
		pantheios::log(pantheios::alert, PSTR("out of memory"));
	}
	catch(std::exception& x)
	{
		pantheios::log_CRITICAL(PSTR("Exception: "), x);
	}
	catch(...)
	{
		pantheios::logputs(pantheios::emergency, PSTR("Unexpected unknown error"));
	}
}

/* TYPEDEFS */
typedef unsigned long long int ullint;
typedef stx::btree<ullint, int> tree;
typedef tree* tree_pointer;
typedef int tag_type;
typedef float store_value_type;
typedef float info_value_type;

#endif /* DDJ_STOREINCLUDES_H_ */
