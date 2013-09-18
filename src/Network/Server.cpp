/*
 * Server.cpp
 *
 *  Created on: Sep 17, 2013
 *      Author: janisz
 */

#include "Server.h"
#include "../Store/StoreIncludes.h"

Server::Server(int port) {
	pantheios::log_INFORMATIONAL("Setting up server! ", "[", boost::lexical_cast<std::string>(boost::this_thread::get_id()), "]");
	pantheios::log_INFORMATIONAL("Port: ", boost::lexical_cast<std::string>(port));
	acceptor = new tcp::acceptor(ioService, tcp::endpoint(tcp::v4(), port));
}

Server::~Server() {
	acceptor->close();
	delete acceptor;
}

void Server::listen()
{
	pantheios::log_INFORMATIONAL("Server is listening...");
	unsigned int numberOfConnections = 0;
	for (;;)
	{
		socket_ptr socket(new tcp::socket(ioService)); //socket per client
		pantheios::log_DEBUG("Creating new socket");
		acceptor->accept(*socket);
		pantheios::log_INFORMATIONAL("New connection accepted [#", boost::lexical_cast<std::string>(numberOfConnections++), "]");
		boost::thread t(boost::bind(&Server::session, this, socket));
	}
}

void Server::session(socket_ptr socket)
{
	pantheios::log_INFORMATIONAL("Session started...");
	try
	{
		//TODO: Implement
		char data[MAX_LENGTH];
		char OK[] = "OK\0";
		boost::system::error_code error;
		socket->read_some(boost::asio::buffer(data), error);
		h_LogThreadInfo(data);
		if (error == boost::asio::error::eof)
			return;
		else if (error)
			throw boost::system::system_error(error); // Some other error.
		boost::asio::write(*socket, boost::asio::buffer(OK, 3));
		socket->close();
	}
	catch (std::exception &e)
	{
		h_LogThreadError(e.what());
	}
}


