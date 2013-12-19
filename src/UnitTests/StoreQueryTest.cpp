#include "StoreQueryTest.h"
namespace ddj {
namespace store {
TEST_F(StoreQueryTest, Constructor)
{
    FAIL() << _storeQuery->toString();
    EXPECT_TRUE(_storeQuery != NULL);
}
}
}
