/*
 * Client.cpp
 *
 *  Created on: Sep 17, 2013
 *      Author: janisz
 */

#include "Client.h"

Client::Client(std::string ip, std::string port) {
	this->host = ip;
	this->port = port;
	socket = new tcp::socket(io_service);
}

void Client::connect()
{
	tcp::resolver resolver(io_service);
	boost::asio::connect(*socket, resolver.resolve({host.c_str(), port.c_str()}));
}

void Client::write(char* message, size_t length)
{
	boost::asio::write(*socket, boost::asio::buffer(message, length));
}

size_t Client::read(char* replay, size_t length)
{
	return boost::asio::read(*socket,
            boost::asio::buffer(replay, length));
}

void Client::close()
{
	boost::system::error_code ec;
	socket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
	if (ec)
	{
		h_LogThreadError("Problem with closing socket");
	}
}

Client::~Client() {
	close();
	delete socket;
}

