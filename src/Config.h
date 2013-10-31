/*
 * Config.h
 *
 *  Created on: Oct 31, 2013
 *      Author: dud
 */

#ifndef CONFIG_H_
#define CONFIG_H_


#include <boost/program_options.hpp>
namespace po = boost::program_options;


#include <iostream>
#include <fstream>
#include <iterator>
using namespace std;



class Config
{
public:
	void ReadFromFile();
	Config();
	virtual ~Config();
};

#endif /* CONFIG_H_ */
