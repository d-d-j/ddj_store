#ifndef NETWORK_CLIENT_H_
#define NETWORK_CLIENT_H_

#include "NetworkLoginRequest.h"
#include "../Task/TaskRequest.h"
#include "../Task/TaskResult.h"
#include "../Store/StoreElement.h"
#include "../Core/Config.h"
#include "../Core/Logger.h"
#include <boost/signals2/signal.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace ddj {
namespace network {

	using boost::asio::ip::tcp;

	class NetworkClient {
	private:
		// LOGGER & CONFIG
		Logger _logger = Logger::getRoot();
		Config* _config = Config::GetInstance();

		// CONNECTION DATA
		std::string host;
		std::string port;

		// NETWORK SERVICE
		boost::asio::io_service io_service;
		tcp::socket *socket;

		// SIGNAL
		boost::signals2::signal<void (task::taskRequest)> *requestSignal;
	public:
		NetworkClient(std::string ip, std::string port);
		NetworkClient(boost::signals2::signal<void (task::taskRequest)>* _requestSignal);
		virtual ~NetworkClient();

		/* SEND METHODS */
		void SendTaskResult(task::taskResult* taskResult);
		void SendLoginRequest(networkLoginRequest* request);

		/* NETWORK CLIENT */
		void connect();
		void write(char* message, size_t length);
		size_t read(char* replay, size_t length);
		void close();

	private:
		void do_read();
	};

} /* namespace store */
} /* namespace network */

#endif /* NETWORK_CLIENT_H_ */
