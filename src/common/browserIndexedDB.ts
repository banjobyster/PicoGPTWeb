const DB_NAME = 'GPT2_Model124M_IndexedDB';
const STORE_NAME = 'paramsStore';

export const loadIntoIndexedDB = (buffer: Float32Array): Promise<string> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);

    request.onerror = (event) => {
      reject('Error opening IndexedDB');
    };

    request.onupgradeneeded = (event) => {
      const db = (event.target as any).result;
      const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      store.put({ id: 1, data: buffer });
    };

    request.onsuccess = (event) => {
      const db = (event.target as any).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        reject('Float32Array buffer not found in IndexedDB');
      }

      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      store.put({ id: 1, data: buffer });

      transaction.oncomplete = () => {
        resolve('Float32Array buffer stored successfully in IndexedDB');
      };

      transaction.onerror = () => {
        reject('Error storing Float32Array buffer in IndexedDB');
      };
    };
  });
};

export const getFromIndexedDB = (): Promise<Float32Array> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);

    request.onerror = (event) => {
      reject('Error opening IndexedDB');
    };

    request.onupgradeneeded = (event) => {
      const db = (event.target as any).result;
      db.createObjectStore(STORE_NAME, { keyPath: 'id' });
    };

    request.onsuccess = async (event) => {
      const db = (event.target as any).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        reject('Float32Array buffer not found in IndexedDB');
      }

      const transaction = db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const getRequest = store.get(1);

      getRequest.onsuccess = (event) => {
        if ((event.target as any).result) {
          resolve((event.target as any).result.data);
        } else {
          reject('Float32Array buffer not found in IndexedDB');
        }
      };

      getRequest.onerror = () => {
        reject('Error retrieving Float32Array buffer from IndexedDB');
      };
    };
  });
};

export const deleteFromIndexedDB = (): Promise<string> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);

    request.onerror = (event) => {
      reject('Error opening IndexedDB');
    };

    request.onsuccess = (event) => {
      const db = (event.target as any).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        reject('Float32Array buffer not found in IndexedDB');
      }

      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const deleteRequest = store.delete(1);

      deleteRequest.onsuccess = () => {
        resolve('Float32Array buffer deleted successfully from IndexedDB');
      };

      deleteRequest.onerror = () => {
        reject('Error deleting Float32Array buffer from IndexedDB');
      };
    };
  });
};

export const checkStoreInIndexedDB = (): Promise<boolean> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);

    request.onerror = (event) => {
      reject('Error opening IndexedDB');
    };

    request.onsuccess = async (event) => {
      const db = (event.target as any).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        reject('Float32Array buffer not found in IndexedDB');
      }
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const objectStore = transaction.objectStore(STORE_NAME);

      const countRequest = objectStore.count(1);

      countRequest.onsuccess = function (event) {
        const count = (event.target as any).result;
        if (count > 0) {
          resolve(true);
        } else {
          reject('Float32Array buffer not found in IndexedDB');
        }
      };

      countRequest.onerror = function (event) {
        reject('Error checking if value exists: ' + event.target.errorCode);
      };
    };
  });
};
