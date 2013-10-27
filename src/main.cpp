/*
 * main.cpp
 *
 *  Created on: Aug 10, 2013
 *      Author: Karol Dzitkowski
 *
 *      NAZEWNICTWO
 * 1. nazwy klas:  CamelStyle z dużej litery np. StoreController
 * 2. nazwy struktur camelStyle z małej litery np. storeElement
 * 3. nazwy pól prywatnych camelStyle z małej litery z podkreśleniem _backBuffer
 * 4. nazwy pól publicznych i zmiennych globalnych słowa rozdzielamy _ i z małych liter np. memory_available
 * 5. define z dużych liter i rozdzielamy _ np. BUFFER_SIZE
 * 6. nazwy metod publicznych z dużej litery CamelStyle np. InsertValue() oraz parametry funkcji z małych liter camelStyle np. InsertValue(int valToInsert);
 * 7. nazwy metod prywatnych z małej litery camelStyle
 * 8. nazwy funkcji "prywatnych" w plikach cpp z małej litery z _ czyli, insert_value(int val_to_insert);
 * 9. nazwy funkcji globalnych czyli w plikach .h najczęściej inline h_InsertValue() dla funkcji na CPU g_InsertValue() dla funkcji na GPU
 */

#include "Node.h"
#include "Store/storeElement.h"
#include "Network/Client.h"

using namespace ddj::store;

const PAN_CHAR_T PANTHEIOS_FE_PROCESS_IDENTITY[] = "DDJ_Store";

int main()
{
	pantheios::init();
	pantheios::log_INFORMATIONAL("Main function started! ", "[Thread id = ", boost::lexical_cast<std::string>(boost::this_thread::get_id()), "]");

	Client c("127.0.0.1", "8080");
	c.connect();

	const int LEN = 100;

	char msg[LEN] = "Test message";

	while (true) {
		c.write(msg, LEN);
		c.read(msg, LEN);
		pantheios::log_INFORMATIONAL("Received message: ", msg );
	}

	return EXIT_SUCCESS;
}


