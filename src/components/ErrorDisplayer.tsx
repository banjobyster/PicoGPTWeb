import { useEffect, useState } from 'react';
import { useStore } from '../store/store';
import { Alert, Fade } from '@mui/material';

const ErrorDisplayer = () => {
  const { error } = useStore();
  const [showError, setShowError] = useState(false);

  useEffect(() => {
    if (error) {
      setShowError(true);
      setTimeout(() => {
        setShowError(false);
      }, 5000);
    }
  }, [error]);

  return (
    <Fade in={showError}>
      <Alert
        severity='error'
        sx={{
          position: 'fixed',
          bottom: 40,
          left: 40,
          minWidth: 200
        }}
      >
        {error}
      </Alert>
    </Fade>
  );
};

export default ErrorDisplayer;
