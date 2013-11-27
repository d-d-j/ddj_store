/*
 * Client.cpp
 *
 *  Created on: Sep 17, 2013
 *      Author: janisz
 */

#include "NetworkClient.h"

namespace ddj {
namespace network {

	NetworkClient::NetworkClient(std::string ip, std::string port)
	{
		this->host = ip;
		this->port = port;
		socket = new tcp::socket(io_service);
	}

	NetworkClient::NetworkClient(boost::signals2::signal<void (task::taskRequest)> *_requestSignal)
		: NetworkClient("127.0.0.1", "8080")
	{
		connect();
		requestSignal = _requestSignal;
		boost::thread workerThread(&NetworkClient::do_read, this);
	}

	void NetworkClient::connect()
	{
		tcp::resolver resolver(io_service);
		boost::asio::connect(*socket, resolver.resolve({host.c_str(), port.c_str()}));
	}

	void NetworkClient::SendTaskResult(task::taskResult* taskResult)
	{
		int len = 3*sizeof(int) + taskResult->result_size;
		char* msg = new char[len];

		memcpy(msg, &(taskResult->task_id), sizeof(int));
		memcpy(msg + sizeof(int), &(taskResult->type), sizeof(int));
		memcpy(msg + 2*sizeof(int), &(taskResult->result_size), sizeof(int));
		memcpy(msg + 3*sizeof(int), taskResult->result_data, taskResult->result_size);

		write(msg, len);
	}

	void NetworkClient::write(char *message, size_t length)
	{
		boost::asio::write(*socket, boost::asio::buffer(message, length));
	}

	void NetworkClient::do_read()
	{
		static int id = 0;
		const int LEN = 100;
		char msg[LEN];

		//TODO: Split reading to read header first and then data
		//TODO: Think about alignment
		while (read(msg, LEN))
		{
			id++;

			task::taskRequest tr;
			store::storeElement* se = new store::storeElement();

			memcpy(&tr, msg, sizeof(tr) - sizeof(void*));
			tr.data = nullptr;
			memcpy(se, msg + sizeof(tr) - sizeof(void*) - 4, sizeof(*se));
			tr.data = se;
			(*requestSignal)(tr);
		}
	}

	size_t NetworkClient::read(char *replay, size_t length)
	{
		return boost::asio::read(*socket,
								 boost::asio::buffer(replay, length));
	}

	void NetworkClient::close()
	{
		boost::system::error_code ec;
		socket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
		if (ec)
		{
		}
	}

	NetworkClient::~NetworkClient()
	{
		close();
		delete socket;
	}

} /* namespace store */
} /* namespace network */
