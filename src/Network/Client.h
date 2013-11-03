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
#include "../Task/TaskRequest.h"
 #include "../Task/TaskResult.h"

using boost::asio::ip::tcp;

class Client {
private:
	std::string host;
	std::string port;
	boost::asio::io_service io_service;
	tcp::socket *socket;
	char msg[100];
	boost::signals2::signal<void (taskRequest)> *requestSignal;
	void do_read();
public:
	Client(std::string ip, std::string port);
	Client(boost::signals2::signal<void (taskRequest)>* _requestSignal);
	virtual ~Client();
	void SendTaskResult(ddj::TaskResult taskResult);
	void connect();
	void write(char* message, size_t length);
	size_t read(char* replay, size_t length);
	void close();
};

#endif /* CLIENT_H_ */
