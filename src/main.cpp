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

#include "Store/StoreController.h"

using namespace ddj::store;

const PAN_CHAR_T PANTHEIOS_FE_PROCESS_IDENTITY[] = "DDJ_Store";

int main()
{
	pantheios::init();
	pantheios::log_INFORMATIONAL("Main function started! ", "[Thread id = ", boost::lexical_cast<std::string>(boost::this_thread::get_id()), "]");
	StoreController* store = new StoreController();

	storeElement e1,e2,e3;
	e1.series = 1;
	e1.tag = 5;
	e1.time = 10;
	e1.value = 1.1;

	e2.series = 1;
	e2.tag = 5;
	e2.time = 12;
	e2.value = 2.2;

	e3.series = 1;
	e3.tag = 5;
	e3.time = 15;
	e3.value = 3.3;

	store->InsertValue(&e1);
	store->InsertValue(&e2);
	store->InsertValue(&e3);

	delete store;

	return EXIT_SUCCESS;
}
