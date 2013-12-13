/*
 * NetworkLogin.h
 *
 *  Created on: 02-12-2013
 *      Author: ghash
 */

#ifndef NETWORKLOGINREQUEST_H_
#define NETWORKLOGINREQUEST_H_

namespace ddj {
namespace network {

struct networkLoginRequest {
public:
	int cuda_devices_count;
	int* devices;

	networkLoginRequest(int* devices, int count)
	{
		this->cuda_devices_count = count;
		this->devices = devices;
	}
};

} /* namespace network */
} /* namespace ddj */
#endif /* NETWORKLOGINREQUEST_H_ */
