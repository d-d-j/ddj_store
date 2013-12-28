/*
 * Client.cpp
 *
 *  Created on: Sep 17, 2013
 *      Author: janisz
 */

#include "NetworkClient.h"
#include "cstdlib"

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
		socket = nullptr;
	}

	void NetworkClient::connect()
	{
		bool connected = false;
		int32_t delay = 1000;
		const int numberOfTryies = 10;
		for (int i=0;i<numberOfTryies && !connected;i++)
		{
			LOG4CPLUS_INFO(this->_logger, "Trying to connect");
			try
			{
				tcp::resolver resolver(io_service);
				boost::asio::connect(*socket, resolver.resolve({host.c_str(), port.c_str()}));
				connected = true;
			}
			catch (boost::system::system_error const& e)
			{
				LOG4CPLUS_ERROR(this->_logger, "Error while trying to connect - " << e.what());
				connected = false;
			}
			boost::this_thread::sleep(boost::posix_time::milliseconds(delay));
		}
		if (!connected)
		{
			LOG4CPLUS_ERROR(this->_logger, "Can't connect with master");
			LOG4CPLUS_INFO(this->_logger, "Program will be terminated");
			std::exit(EXIT_FAILURE);
		}
		LOG4CPLUS_INFO(this->_logger, "Connection established");
	}

	void NetworkClient::SendLoginRequest(networkLoginRequest* request)
	{
		size_t len = sizeof(int)*(request->cuda_devices_count + 1);
		char msg[len];

		memcpy(msg, &(request->cuda_devices_count), sizeof(int));
		memcpy(msg + sizeof(int), request->devices, sizeof(int)*request->cuda_devices_count);

		write(msg, len);
	}

	void NetworkClient::SendTaskResult(task::taskResult* taskResult)
	{
		int deviceId = TASK_ALL_DEVICES;
		int len = sizeof(int64_t) + 3*sizeof(int32_t) + taskResult->result_size;
		char oldMsg[len];
		char *msg = oldMsg;
		memcpy(msg, &(taskResult->task_id), sizeof(int64_t));
		msg += sizeof(int64_t);
		memcpy(msg, &(taskResult->type), sizeof(int32_t));
		msg += sizeof(int32_t);
		memcpy(msg, &(taskResult->result_size), sizeof(int32_t));
		msg += sizeof(int32_t);
		memcpy(msg, &deviceId, sizeof(int32_t));
		msg += sizeof(int32_t);
		memcpy(msg, taskResult->result_data, taskResult->result_size);

		write(oldMsg, len);
	}

	void NetworkClient::write(char *message, size_t length)
	{
		boost::asio::write(*socket, boost::asio::buffer(message, length));
	}

	void NetworkClient::do_read()
	{
		const int LEN = sizeof(int32_t)*3 + sizeof(int64_t);
		char msg[LEN];

		while (read(msg, LEN))
		{
			task::taskRequest tr;

			// COPY HEADER
			int position = 0;
			memcpy(&tr.task_id, msg+position, sizeof(int64_t));
			position+=sizeof(int64_t);
			memcpy(&tr.type, msg+position, sizeof(int32_t));
			position+=sizeof(int32_t);
			memcpy(&tr.size, msg+position, sizeof(int32_t));
			position+=sizeof(int32_t);
			memcpy(&tr.device_id, msg+position, sizeof(int32_t));
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
			LOG4CPLUS_INFO(this->_logger, "Program will be terminated");
			std::exit(EXIT_FAILURE);
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
