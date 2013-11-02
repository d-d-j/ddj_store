/*
 * Client.cpp
 *
 *  Created on: Sep 17, 2013
 *      Author: janisz
 */

#include "Client.h"
#include "../Store/LoggerHelper.h"
#include "../Store/StoreIncludes.h"
#include "../Store/storeElement.h"
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <cstdlib>

Client::Client(std::string ip, std::string port)
{
    this->host = ip;
    this->port = port;
    socket = new tcp::socket(io_service);
}

Client::Client(boost::signals2::signal<void (taskRequest)> *_requestSignal)
    : Client("127.0.0.1", "8080")
{
    connect();
    requestSignal = _requestSignal;
    h_LogThreadDebug("Introduce to server");
    char msg[] = "Node #1";
    write(msg, strlen(msg));
    boost::thread workerThread(&Client::do_read, this);

}

void Client::connect()
{
    h_LogThreadDebug(("Connecting with " + host + ":" + port + " ...").c_str());
    tcp::resolver resolver(io_service);
    boost::asio::connect(*socket, resolver.resolve({host.c_str(), port.c_str()}));
    h_LogThreadDebug("Connection established");
}

void Client::write(char *message, size_t length)
{
    h_LogThreadDebug("Sending message...");
    boost::asio::write(*socket, boost::asio::buffer(message, length));
}

void Client::do_read()
{
    static int id = 0;

    const int LEN = 100;
    char msg[LEN];

    //TODO: Split reading to read header first and then data
    //TODO: Think about alignment
    while (int n = read(msg, LEN))
    {
        id++;
        h_LogThreadDebug("Input: ");

        taskRequest tr;
        ddj::store::storeElement se;

        memcpy(&tr, msg, sizeof(tr) - sizeof(void*));
        tr.data = nullptr;
        h_LogThreadDebug(tr.toString().c_str());
        memcpy(&se, msg + sizeof(tr) - sizeof(void*) - 4, sizeof(se));
        tr.data = &se;
        h_LogThreadDebug(se.toString().c_str());
        (*requestSignal)(tr);
    }
}

size_t Client::read(char *replay, size_t length)
{
    h_LogThreadDebug("Reading message...");

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

Client::~Client()
{
    h_LogThreadDebug("Client destructor fired");
    close();
    delete socket;
}

