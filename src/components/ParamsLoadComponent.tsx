import { useEffect, useState } from 'react';
import { useStore } from '../store/store';
import { loadParamsFromGPT2 } from '../common/paramsExtractor';
import { deleteFromIndexedDB } from '../common/browserIndexedDB';
import {
  Alert,
  Box,
  Button,
  LinearProgress,
  LinearProgressProps,
  Paper,
  Tooltip,
  Typography
} from '@mui/material';
import { checkStoreInIndexedDB } from '../common/browserIndexedDB';
import { getGpu } from '../common/gpuUtils';

const LinearProgressWithLabel = (props: LinearProgressProps & { value: number }) => {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
      <Box sx={{ width: '100%', mr: 1 }}>
        <LinearProgress variant='determinate' {...props} />
      </Box>
      <Box sx={{ minWidth: 105 }}>
        <Typography variant='body2' color='text.secondary'>
          {`${Math.floor(props.value * 4.98)}MB`} / 498MB
        </Typography>
      </Box>
    </Box>
  );
};

const ParamsLoadComponent = () => {
  const [loaded, setLoaded] = useState(false);
  const [downloaded, setDownloaded] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [gpuFound, setGpuFound] = useState(true);
  const { setParams, setError } = useStore();

  const handleLoad = () => {
    (async () => {
      try {
        const params = await loadParamsFromGPT2(setDownloadProgress);
        setParams(params);
        setLoaded(true);
        setDownloaded(true);
      } catch (error) {
        setError(error.message);
      }
    })();
  };

  const handleUnload = () => {
    setLoaded(false);
    (async () => {
      try {
        await deleteFromIndexedDB();
        setParams({});
      } catch (error) {
        setError(error.message);
      }
    })();
  };

  useEffect(() => {
    (async () => {
      try {
        await checkStoreInIndexedDB();
        setDownloaded(true);
      } catch (error) {
        setDownloaded(false);
      }
    })();
    (async () => {
      try {
        await getGpu();
        setGpuFound(true);
      } catch (error) {
        setError(error.message);
        setGpuFound(false);
      }
    })();
  }, []);

  return (
    <Paper
      style={{
        marginBottom: '10px',
        minWidth: '100px',
        color: '#0C134F',
        fontWeight: 'bold',
        padding: '10px'
      }}
    >
      <div
        style={{
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-around',
          flexWrap: 'wrap',
          gap: '10px'
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'row',
            gap: '10px',
            flexWrap: 'wrap'
          }}
        >
          <Tooltip title='Load model from browser storage'>
            <Button onClick={handleLoad} variant='outlined' disabled={loaded || !downloaded}>
              Load Model
            </Button>
          </Tooltip>
          <Tooltip title='Delete model from browser storage'>
            <Button onClick={handleUnload} variant='outlined' disabled={!downloaded}>
              Delete Model
            </Button>
          </Tooltip>
          {!gpuFound && <Alert severity='error'>WebGPU not found. Running on CPU.</Alert>}
        </div>
        <div
          style={{
            display: 'flex',
            flexDirection: 'row',
            gap: '10px',
            flexWrap: 'wrap'
          }}
        >
          <Button onClick={handleLoad} variant='outlined' disabled={downloaded}>
            Download Model
          </Button>
        </div>
      </div>

      {!downloaded && downloadProgress > 0 && (
        <LinearProgressWithLabel
          variant='determinate'
          value={downloadProgress}
          style={{ width: '100%' }}
        />
      )}
    </Paper>
  );
};

export default ParamsLoadComponent;
