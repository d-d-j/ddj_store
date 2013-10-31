/*
 * Config.h
 *
 *  Created on: Oct 31, 2013
 *      Author: dud
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include <iostream>
#include <fstream>
#include <iterator>

#include <boost/program_options.hpp>

#include "Store/LoggerHelper.h"
namespace po = boost::program_options;

using namespace std;

class Config
{
private:
	static Config* _instance;

	po::variables_map _configMap;
	Config();
	virtual ~Config();

public:
	static Config* GetInstance();
	void ListAllSettings();
};

#endif /* CONFIG_H_ */
