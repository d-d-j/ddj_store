#ifndef DDJ_STORELOGGER_H_
#define DDJ_STORELOGGER_H_

#include "StoreIncludes.h"

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

inline void h_LogThreadWithTagDebug(const char* text, tag_type tag)
{
	if(typeid(tag_type) != typeid(int))
	{
		h_LogThreadDebug(text);
		return;
	}

	pantheios::integer iTag((int)tag);

	try
	{
		pantheios::log_DEBUG(
				PSTR("[Thread id="),
				boost::lexical_cast<std::string>(boost::this_thread::get_id()),
				PSTR("] "),
				PSTR(text),
				PSTR(" [Tag="),
				iTag,
				PSTR("]"));
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

#endif /* DDJ_STORELOGGER_H_ */
