#ifndef NETWORK_CLIENT_H_
#define NETWORK_CLIENT_H_

#include "../Task/TaskRequest.h"
#include "../Task/TaskResult.h"
#include "../Store/StoreElement.h"
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
		std::string host;
		std::string port;
		boost::asio::io_service io_service;
		tcp::socket *socket;
		boost::signals2::signal<void (task::taskRequest)> *requestSignal;
		void do_read();
	public:
		NetworkClient(std::string ip, std::string port);
		NetworkClient(boost::signals2::signal<void (task::taskRequest)>* _requestSignal);
		virtual ~NetworkClient();
		void SendTaskResult(task::taskResult* taskResult);
		void connect();
		void write(char* message, size_t length);
		size_t read(char* replay, size_t length);
		void close();
	};

} /* namespace store */
} /* namespace network */

#endif /* NETWORK_CLIENT_H_ */
