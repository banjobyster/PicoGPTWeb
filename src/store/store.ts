import { create } from 'zustand';

export interface StoreInterface {
  params: object | null;
  setParams: (params: object) => void;
  error: string | null;
  setError: (error: string) => void;
}

export const useStore = create<StoreInterface>((set) => ({
  params: null,
  setParams: (params: object) => set({ params }),
  error: null,
  setError: (error: string) => set({ error })
}));
