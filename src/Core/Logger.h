/*
 * Logger.h
 *
 *  Created on: 06-11-2013
 *      Author: ghash
 */

#ifndef LOGGER_H_
#define LOGGER_H_

#include <log4cplus/loggingmacros.h>
#include <log4cplus/configurator.h>
#include <log4cplus/helpers/loglog.h>
#include <log4cplus/helpers/stringhelper.h>
#include <log4cplus/loggingmacros.h>
#include <log4cplus/hierarchy.h>


using namespace log4cplus;
using namespace log4cplus::helpers;

/* USAGE:
 * 	LOG4CPLUS_FATAL(root, LOG4CPLUS_TEXT("Fatal...."));
 *	LOG4CPLUS_ERROR(root, LOG4CPLUS_TEXT("Error...."));
 *	LOG4CPLUS_WARN(root, LOG4CPLUS_TEXT("Warn...."));
 *	LOG4CPLUS_INFO(root, LOG4CPLUS_TEXT("Info...."));
 *	LOG4CPLUS_DEBUG(root, LOG4CPLUS_TEXT("Debug...."));
 *  OR LIKE THIS
 *  LOG4CPLUS_INFO_FMT(_logger, "Insert element[ tag=%d, metric=%d, time=%d, value=%f", element->tag, element->series, element->time, element->value);
 *  OR LIKE THIS
 *  LOG4CPLUS_INFO(_logger, "Insert: t=" << element->tag << " s=" << element->series << " t=" << element->time << "v=" << element->value);
 */

#endif /* LOGGER_H_ */
