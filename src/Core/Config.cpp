/*
 * Config.cpp
 *
 *  Created on: Oct 31, 2013
 *      Author: dud
 */

#include "Config.h"

namespace ddj
{

	Config* Config::_instance(0);

	Config* Config::GetInstance()
	{
		if (!_instance)
		{
			_instance = new Config;
		}
		return _instance;
	}

	int Config::GetIntValue(string settingName)
	{
		if (_configMap.count(settingName))
		{
			return _configMap[settingName].as<int>();
		}
		//LOG4CPLUS_ERROR(this->_logger, LOG4CPLUS_TEXT("value for the setting not found"));
		return -1;
	}

	string Config::GetStringValue(string settingName)
	{
		if (_configMap.count(settingName))
		{
			return _configMap[settingName].as<string>();
		}
		//LOG4CPLUS_ERROR(this->_logger, LOG4CPLUS_TEXT("value for the setting not found"));
		return nullptr;
	}

	void Config::ListAllSettings()
	{
		po::variables_map::iterator it;
		for (it = _configMap.begin(); it != _configMap.end(); ++it)
		{
			cout << it->first << "\n";
		}
	}

	Config::Config()
	{
		try
		{
			string config_file = "config.ini";

			_configMap = po::variables_map();

			po::options_description hidden("Hidden options");
			hidden.add_options()
					("MB_SIZE_IN_BYTES", po::value<int>()->default_value(1048576), "size in bytes")
					("MAIN_STORE_SIZE", po::value<int>()->default_value(512*1048576), "main store size")
					("GPU_MEMORY_ALLOC_ATTEMPTS", po::value<int>()->default_value(8), "number of GPU memory allocation attempts")
					("STREAMS_NUM_QUERY", po::value<int>()->default_value(2), "number of query streams")
					("STREAMS_NUM_UPLOAD", po::value<int>()->default_value(2), "number of upload streams")
					("STORE_BUFFER_CAPACITY", po::value<int>()->default_value(512), "store buffer capacity")
					("THREAD_POOL_SIZE", po::value<int>()->default_value(1000), "number of threads in thread pool for tasks")
					("SIMUL_QUERY_COUNT", po::value<int>()->default_value(3), "number of simultaneous queries")
					("MASTER_IP_ADDRESS", po::value<string>()->default_value("127.0.0.1"), "address of master server")
					("MASTER_LOGIN_PORT", po::value<string>()->default_value("8080"), "port of master server login service");


			po::options_description config_file_options;
			config_file_options.add(hidden);

			ifstream ifs(config_file.c_str());
			if (!ifs)
			{
				string msg = "can not open config file: ";
				msg.append(config_file);
				//LOG4CPLUS_ERROR(this->_logger, LOG4CPLUS_TEXT(msg.c_str()));
				return;
			}
			else
			{
				store(parse_config_file(ifs, config_file_options), _configMap);
				notify(_configMap);
			}

			//LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Finished loading settings from file"));

		} catch (exception& e)
		{
			//LOG4CPLUS_ERROR(this->_logger, LOG4CPLUS_TEXT(e.what()));
			return;
		}
		return;
	}

	Config::~Config()
	{
		// TODO Auto-generated destructor stub
	}


} /* namespace ddj */
