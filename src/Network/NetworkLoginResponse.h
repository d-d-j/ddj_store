#ifndef NETWORKLOGINRESPONSE_H_
#define NETWORKLOGINRESPONSE_H_

namespace ddj {
namespace network {

struct networkLoginResponse {
public:
	int node_id;

	networkLoginResponse(int id)
	{
		node_id = id;
	}
};

} /* namespace network */
} /* namespace ddj */
#endif /* NETWORKLOGINRESPONSE_H_ */
