/*
 * Client.h
 *
 *  Created on: Sep 17, 2013
 *      Author: janisz
 */

#ifndef CLIENT_H_
#define CLIENT_H_

#include <boost/asio.hpp>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <boost/asio.hpp>
#include "../Store/StoreIncludes.h"

using boost::asio::ip::tcp;

class Client {
private:
	std::string host;
	std::string port;
	boost::asio::io_service io_service;
	tcp::socket *socket;
public:
	Client(std::string ip, std::string port);
	virtual ~Client();
	void connect();
	void write(char* message, size_t length);
	size_t read(char* replay, size_t length);
	void close();
};

#endif /* CLIENT_H_ */
