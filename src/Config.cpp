/*
 * Config.cpp
 *
 *  Created on: Oct 31, 2013
 *      Author: dud
 */

#include "Config.h"

void Config::ReadFromFile()
{
	try
	{
		string config_file = "config.ini";

//		// Declare a group of options that will be
//		// allowed only on command line
//		po::options_description generic("Generic options");
//		generic.add_options()("version,v", "print version string")("help",
//				"produce help message")("config,c",
//				po::value < string
//						> (&config_file)->default_value("multiple_sources.cfg"),
//				"name of a file of a configuration.");

// Declare a group of options that will be

// Hidden options, will be allowed both on command line and
// in config file, but will not be shown to the user.
		po::options_description hidden("Hidden options");
		hidden.add_options()("MB_SIZE_IN_BYTES",
				po::value<int>()->default_value(1048576), "size in bytes")(
				"STORE_BUFFER_SIZE", po::value<int>()->default_value(2),
				"store buffer size")("MAIN_STORE_SIZE",
				po::value<int>()->default_value(536870912), "main store size")(
				"GPU_MEMORY_ALLOC_ATTEMPTS", po::value<int>()->default_value(4),
				"number of GPU memory allocation attempts")("STREAMS_NUM_UPLOAD",
				po::value<int>()->default_value(2), "number of upload streams")(
				"STREAMS_NUM_QUERY", po::value<int>()->default_value(2),
				"number of query streams")
				("DEVICE_BUFFERS_COUNT", po::value<int>()->default_value(1),
				"number of device buffers")	("SIMUL_QUERY_COUNT", po::value<int>()->default_value(3),
				"number of simultaneous queries");

//		po::options_description cmdline_options;
//		cmdline_options.add(generic).add(config).add(hidden);

		po::options_description config_file_options;
		config_file_options.add(hidden);

//		po::options_description visible("Allowed options");
//		visible.add(generic).add(config);

//		po::positional_options_description p;
//		p.add("input-file", -1);

		po::variables_map vm;
//		store(
//				po::command_line_parser(ac, av).options(cmdline_options).positional(
//						p).run(), vm);
//		notify(vm);

		ifstream ifs(config_file.c_str());
		if (!ifs)
		{
			cout << "can not open config file: " << config_file << "\n";
			return;
		}
		else
		{
			store(parse_config_file(ifs, config_file_options), vm);
			notify(vm);
		}

//		po::variables_map::iterator it;
//		for (it = vm.begin(); it != vm.end(); ++it)
//		{
//			cout << it->first << "=" <<  "\n";
//		}

	} catch (exception& e)
	{
		cout << e.what() << "\n";
		return;
	}
	return;
}

Config::Config()
{

}

Config::~Config()
{
	// TODO Auto-generated destructor stub
}

