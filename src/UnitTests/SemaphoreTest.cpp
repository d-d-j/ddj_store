#include "SemaphoreTest.h"

namespace ddj {
namespace unittest {

	TEST_F(SemaphoreTest, Constructor)
	{
		EXPECT_TRUE(_semaphore != NULL);
	}

	TEST_F(SemaphoreTest, Two_Threads_Trying_To_Acces_Critical_Section)
	{
		int x = 0;
		_semaphore->Wait();
		auto f = [&](){ for (int i=0;i<10;i++) { _semaphore->Wait(); x++; _semaphore->Release();}};
		boost::thread t1(f);
		boost::thread t2(f);
		x = 100*x;

		EXPECT_EQ(0, x);

		_semaphore->Release();
	    t1.join();
	    t2.join();

		EXPECT_EQ(20, x);
	}

	TEST_F(SemaphoreTest, Three_Threads_Trying_To_Acces_Critical_Section_Sempoher_Accept_Only_Two)
	{
		delete _semaphore;
		_semaphore = new Semaphore(2);
		int x = 0, y = 0;
		_semaphore->Wait();

		auto f1 = [&](){ _semaphore->Wait(); x++; _semaphore->Release();};
		auto f2 = [&](){ _semaphore->Wait(); y++; _semaphore->Release();};
		boost::thread t1(f1);
		boost::thread t2(f2);

		EXPECT_EQ(0, x*y);
		EXPECT_EQ(1, x+y);

		_semaphore->Release();
	    t1.join();
	    t2.join();

		EXPECT_EQ(1, x);
		EXPECT_EQ(1, y);
	}
}
}
