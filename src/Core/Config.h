/*
 * Config.h
 *
 *  Created on: Oct 31, 2013
 *      Author: dud
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include "Logger.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

namespace ddj
{
	class Config
	{
	private:
		static Config* _instance;

		po::variables_map _configMap;

		//Logger _logger = Logger::getRoot();
		Config();
		virtual ~Config();

	public:
		int GetIntValue(string);
		string GetStringValue(string settingName);
		static Config* GetInstance();
		void ListAllSettings();
	};
} /* namespace ddj */
#endif /* CONFIG_H_ */
