import ErrorDisplayer from './components/ErrorDisplayer.tsx';
import InferenceWindow from './components/InferenceWindow.tsx';
import ParamsLoadComponent from './components/ParamsLoadComponent.tsx';

const App = () => {
  return (
    <div>
      <ParamsLoadComponent />
      <InferenceWindow />
      <ErrorDisplayer />
    </div>
  );
};

export default App;
