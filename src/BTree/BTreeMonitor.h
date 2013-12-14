#ifndef BTREEMONITOR_H_
#define BTREEMONITOR_H_

#include "btree.h"
#include "../Core/Logger.h"
#include "../Store/StoreTrunkInfo.h"
#include <boost/thread.hpp>
#include <boost/container/vector.hpp>
#include <stdexcept>

namespace ddj {
namespace btree {

	/* TYPEDEFS */
	typedef struct trunkInfo
	{
		ullint start;
		ullint end;

		trunkInfo(){}
		trunkInfo(ullint s, ullint e)
		{
			start = s;
			end = e;
		}
		trunkInfo(const trunkInfo& cp)
		{
			start = cp.start;
			end = cp.end;
		}
		~trunkInfo(){}
	} trunkInfo;
	typedef struct timePeriod
	{
		ullint start;
		ullint end;

		bool operator< (const timePeriod& rhs) const
		{
			if(end < rhs.end) return true;
			else return false;
		}

		bool operator== (const timePeriod& rhs) const
		{
			if(start == rhs.start && end == rhs.end) return true;
			else return false;
		}
	} timePeriod;
	typedef stx::btree<timePeriod, trunkInfo> tree;
	typedef tree* tree_pointer;

	/*! \class BTreeMonitor
	 \brief Protects concurrent access to B+Tree structure using boost::mutex
	*/
	class BTreeMonitor
	{
		private:
			metric_type _metric;				/**< A metric of elements which locations are stored in B+Tree */
			boost::mutex _mutex;				/**< Mutex used to protect access to b+tree */
			tree_pointer _bufferInfoTree;		/**< A pointer to B+Tree structure */

			/* LOGGER */
			Logger _logger = Logger::getRoot();
		public:
			//! BTreeMonitor constructor.
			/*!
			  It sets creates a new instance of B+Tree structure in order to store there
			  additional info about trunk's location in GPU's store array
			*/
			BTreeMonitor(metric_type metric);

			//! BTreeMonitor destructor.
			/*!
			  releases B+Tree structure
			*/
			virtual ~BTreeMonitor();

			/*! \fn  void Insert(infoElement* element);
			 * \brief Function inserts infoElement to B+Tree structure
			 * \param infoElement* a pointer to infoElement to insert to B+Tree structure
			 */
			void Insert(store::storeTrunkInfo* element);

			boost::container::vector<trunkInfo>* SelectAll();
			//boost::container::vector<trunkInfo>* Select(boost::container::vector<timePeriod> timePeriods);
	};

} /* namespace store */
} /* namespace btree */
#endif /* BTREEMONITOR_H_ */
