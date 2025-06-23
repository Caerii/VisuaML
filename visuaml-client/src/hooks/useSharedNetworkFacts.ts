/** @fileoverview This hook synchronizes network metadata from the shared Yjs document to the local Zustand store. */
import { useEffect } from 'react';
import { useYDoc } from '../y/DocProvider';
import { useNetworkStore, type NetworkFacts } from '../store/networkStore';

export const useSharedNetworkFacts = () => {
  const { ydoc } = useYDoc();
  const setFacts = useNetworkStore((state) => state.setFacts);

  useEffect(() => {
    const ySharedFacts = ydoc.getMap('sharedNetworkFacts');

    const syncFacts = () => {
      const facts = Object.fromEntries(ySharedFacts.entries()) as unknown as NetworkFacts;
      setFacts(facts);
    };

    // Initial sync
    syncFacts();

    // Listen for changes
    ySharedFacts.observe(syncFacts);

    return () => {
      ySharedFacts.unobserve(syncFacts);
    };
  }, [ydoc, setFacts]);
}; 