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

namespace ddj {
namespace store {

	/*! \class BTreeMonitor
	 \brief A monitor controlling and managing inserts to B+Tree structure by separate thread.

	 _threadBufferTree thread is started in BTreeMonitor constructor. InfoElements can be set
	  to insert to B+Tree structure by Insert method. It sets _infoElementToInsert to an instance
	  of infoElement to insert, then it notifies _threadBufferTree using _condBufferInfoTree
	  that it has something to insert. And _threadBufferTree does the JOB.
	*/
	class BTreeMonitor
	{
		private:
			boost::thread* _threadBufferTree;	/**< Thread with the aim of inserting infoElements to B+Tree structure */

			boost::condition_variable _condBufferInfoTree;	/**< Condition variable on which _threadBufferTree sleeps */
			boost::condition_variable _condTreeBufferFree;	/**< Condition variable on which BTreeMonitor sleeps when _threadBufferTree is occupied */
			boost::condition_variable _condSynchronization; /**< BTreeMonitor sleeps on it when _threadBufferTree is starting */

			boost::mutex _mutexSynchronization;	/**< Mutex used to synchronize thread creation */
			boost::mutex _mutexBufferInfoTree;	/**< Mutex used by _threadBufferTree and inserts */

			volatile sig_atomic_t _threadBufferTreeStop;	/**< Whether _threadBufferTree is working or not  */
			volatile sig_atomic_t _treeBufferThreadBusy;	/**< Whether _threadBufferTree is busy or not  */

			//TODO: Change _infoElementToInsert to queue of infoElements
			infoElement* _infoElementToInsert;	/**< Elements to insert to B+Tree structure */
			tree_pointer _bufferInfoTree;		/**< A pointer to B+Tree structure */

		public:
			//! BTreeMonitor constructor.
			/*!
			  It sets _bufferInfoTree to specified tree_pointer (tree) and starts a _threadBufferTree thread
			*/
			BTreeMonitor(tree_pointer tree);

			//! BTreeMonitor destructor.
			/*!
			  stops a _threadBufferTree thread
			*/
			virtual ~BTreeMonitor();

			/*! \fn  void Insert(infoElement* element);
			 * \brief Function sets infoElement to insert to B+Tree (it will be inserted there by separate thread as fast as possible)
			 * \param infoElement* a pointer to infoElement to insert to B+Tree structure
			 */
			void Insert(infoElement* element);
		private:
			void threadFunction();	//!< a function which is being executed by _threadBufferTree thread
			bool insertToTree(infoElement* element);	//!< a function which actually inserts element to B+Tree structure
			void startThread();		//!< starts a _threadBufferTree thread
			void stopThread();		//!< stops a _threadBufferTree thread
	};

} /* namespace store */
} /* namespace ddj */
#endif /* BTREEMONITOR_H_ */
