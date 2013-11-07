/*
 * BTreeMonitor.h
 *
 *  Created on: Aug 24, 2013
 *      Author: Karol Dzitkowski
 */

#ifndef BTREEMONITOR_H_
#define BTREEMONITOR_H_

#include "../Store/StoreIncludes.h"
#include "../Store/infoElement.h"
#include "../Helpers/Logger.h"

namespace ddj {
namespace store {

	/*! \class BTreeMonitor
	 \brief Protects concurrent access to B+Tree structure using boost::mutex
	*/
	class BTreeMonitor
	{
		private:
			metric_type _metric;	/**< A metric of elements which locations are stored in B+Tree */
			boost::mutex _mutex;	/**< Mutex used to protect access to b+tree */
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
			void Insert(infoElement* element);

		private:
			void insertToTree(infoElement* element);
	};

} /* namespace store */
} /* namespace ddj */
#endif /* BTREEMONITOR_H_ */
