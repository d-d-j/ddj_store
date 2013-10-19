/*
 * Server.h
 *
 *  Created on: Sep 17, 2013
 *      Author: janisz
 */

#ifndef SERVER_H_
#define SERVER_H_

#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>

using boost::asio::ip::tcp;
typedef boost::shared_ptr<tcp::socket> socket_ptr;

class Server {
private:
	tcp::acceptor* acceptor;
	boost::asio::io_service ioService;
	static const int MAX_LENGTH = 1024;

	void session(socket_ptr socket);

public:
	Server(int port);
	virtual ~Server();
	void listen();

};

#endif /* SERVER_H_ */
