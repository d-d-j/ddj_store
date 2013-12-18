/*
 * Client.cpp
 *
 *  Created on: Sep 17, 2013
 *      Author: janisz
 */

#include "NetworkClient.h"

namespace ddj {
namespace network {

	NetworkClient::NetworkClient(boost::signals2::signal<void (task::taskRequest)> *_requestSignal)
		: _logger(Logger::getRoot()), _config(Config::GetInstance())
	{
		this->host = this->_config->GetStringValue("MASTER_IP_ADDRESS");
		this->port = this->_config->GetStringValue("MASTER_LOGIN_PORT");
		socket = new tcp::socket(io_service);
		requestSignal = _requestSignal;
		connect();
		boost::thread workerThread(&NetworkClient::do_read, this);
	}

	NetworkClient::~NetworkClient()
	{
		close();
		delete socket;
	}

	void NetworkClient::connect()
	{
		tcp::resolver resolver(io_service);
		boost::asio::connect(*socket, resolver.resolve({host.c_str(), port.c_str()}));
	}

	void NetworkClient::SendLoginRequest(networkLoginRequest* request)
	{
		size_t len = sizeof(int)*(request->cuda_devices_count + 1);
		char* msg = new char[len];

		memcpy(msg, &(request->cuda_devices_count), sizeof(int));
		memcpy(msg + sizeof(int), request->devices, sizeof(int)*request->cuda_devices_count);

		write(msg, len);
		delete [] msg;
	}

	void NetworkClient::SendTaskResult(task::taskResult* taskResult)
	{
		int len = sizeof(int64_t) + 2*sizeof(int32_t) + taskResult->result_size;
		LOG4CPLUS_ERROR(this->_logger, "Trying to send " << len);
		char* msg = new char[len];
		char *oldMsg = msg;
		memcpy(msg, &(taskResult->task_id), sizeof(int64_t));
		msg += sizeof(int64_t);
		memcpy(msg, &(taskResult->type), sizeof(int32_t));
		msg += sizeof(int32_t);
		memcpy(msg, &(taskResult->result_size), sizeof(int32_t));
		msg += sizeof(int32_t);
		memcpy(msg, taskResult->result_data, taskResult->result_size);

		write(oldMsg, len);
		delete [] oldMsg;
	}

	void NetworkClient::write(char *message, size_t length)
	{
		boost::asio::write(*socket, boost::asio::buffer(message, length));
	}

	void NetworkClient::do_read()
	{
		const int LEN = sizeof(int32_t)*2 + sizeof(int64_t);
		char msg[LEN];

		while (int readedBytes = read(msg, LEN))
		{
			LOG4CPLUS_DEBUG(this->_logger, "Received: " << readedBytes);
			task::taskRequest tr;

			// COPY HEADER
			memcpy(&tr.task_id, msg, sizeof(int32_t));
			memcpy(&tr.type, msg+sizeof(int64_t), sizeof(int32_t));
			memcpy(&tr.size, msg+(sizeof(int64_t)+sizeof(int32_t)), sizeof(int32_t));
			tr.data = nullptr;

			// READ DATA
			if(tr.size != 0)
			{
				char* dataMsg = new char[tr.size];
				size_t bytesRead = read(dataMsg, tr.size);
				if((int)bytesRead != tr.size)
					LOG4CPLUS_ERROR(this->_logger, "Wrong number of bytes transfered - transfered " << bytesRead << " from " << tr.size);
				tr.data = dataMsg;
			}

			// SIGNAL DATA
			(*requestSignal)(tr);
		}
	}

	size_t NetworkClient::read(char *replay, size_t length)
	{
		try
		{
			return boost::asio::read(*socket,
								 boost::asio::buffer(replay, length));
		}
		catch (const std::exception &e) {
			LOG4CPLUS_ERROR(this->_logger, "Error while reading from socket - " << e.what());
		}
		return 0;
	}

	void NetworkClient::close()
	{
		boost::system::error_code ec;
		socket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
		if (ec)
		{
			LOG4CPLUS_ERROR(this->_logger, "Error while closing network connection - " << ec.message());
		}
	}

} /* namespace store */
} /* namespace network */
