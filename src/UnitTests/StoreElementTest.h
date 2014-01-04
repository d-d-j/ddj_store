#ifndef STORE_ELEMENT_TEST_H_
#define STORE_ELEMENT_TEST_H_

#include "../Store/StoreElement.cuh"
#include <gtest/gtest.h>

namespace ddj {
namespace store {

	class StoreElementTest : public testing::Test {
	protected:
		virtual void SetUp()
		{
			_element = new storeElement(1, 2, 2000, 0.33);
		}

		virtual void TearDown() {
    		delete _element;
		}

		storeElement* _element;
	};

}}	// ddj
#endif // STORE_ELEMENT_TEST_H_
