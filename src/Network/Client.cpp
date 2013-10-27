/*
 * Client.cpp
 *
 *  Created on: Sep 17, 2013
 *      Author: janisz
 */

#include "Client.h"
#include "../Store/LoggerHelper.h"
#include "../Store/StoreIncludes.h"

Client::Client(std::string ip, std::string port) {
	this->host = ip;
	this->port = port;
	socket = new tcp::socket(io_service);
}

void Client::connect()
{
	h_LogThreadDebug(("Connecting with " + host + ":" + port + " ...").c_str());
	tcp::resolver resolver(io_service);
	boost::asio::connect(*socket, resolver.resolve({host.c_str(), port.c_str()}));
	h_LogThreadDebug("Connection established");
}

void Client::write(char* message, size_t length)
{
	h_LogThreadDebug("Sending message...");
	boost::asio::write(*socket, boost::asio::buffer(message, length));
}

size_t Client::read(char* replay, size_t length)
{
	h_LogThreadDebug("Reading message...");
	size_t available = socket -> available();
	length = std::min(available, length);
	return boost::asio::read(*socket,
            boost::asio::buffer(replay, length));
}

void Client::close()
{
	h_LogThreadDebug("Closing connection...");
	boost::system::error_code ec;
	socket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
	if (ec)
	{
		h_LogThreadError("Problem with closing socket");
	}
	h_LogThreadDebug("Connection closed");
}

Client::~Client() {
	h_LogThreadDebug("Client destructor fired");
	close();
	delete socket;
}

