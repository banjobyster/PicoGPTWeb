import { useState, useRef } from 'react';
import { tokenizer } from '../common/bpeTokenizer';
import hparams from '../model124M/hparams';
import { GPT2 } from '../model124M/generate';
import { useStore } from '../store/store';
import { TextareaAutosize } from '@mui/base';
import { Box, Button } from '@mui/material';
import TransformerVisualizer from './TransformerVisualizer';

const TOKEN_LIMIT = 50; // Setting less to avoid memory issues

const InferenceWindow = () => {
  const { params, setError } = useStore();
  const [text, setText] = useState({
    prompt: 'The biggest regret for Dumbledore',
    completion: '',
    lastToken: ''
  });
  const run = useRef(false);
  const [runState, setRunState] = useState(false);
  const [textAreaVisible, setTextAreaVisibility] = useState(true);
  const [n_layer, setNLayer] = useState(0);
  const [topKTokens, setTopKTokens] = useState(['', '', '', '', '']);

  const updateNetworkState = (props) => {
    const { currentLayer, topKTokens } = props;
    if (currentLayer) {
      setNLayer(currentLayer);
    }
    if (topKTokens) {
      setTopKTokens(topKTokens.map((token: number) => tokenizer.decode([token])));
    }
  };

  const generate = async (init: boolean, input: string) => {
    try {
      setTextAreaVisibility(false);
      if (init) {
        run.current = true;
        setRunState(true);
      }
      if (params && run.current) {
        const inputs = tokenizer.encode(input);
        const n_head = hparams.n_head;
        const limitedInputs = inputs.length > TOKEN_LIMIT ? inputs.slice(-TOKEN_LIMIT) : inputs;
        const result = await GPT2.generate(limitedInputs, params, n_head, updateNetworkState);
        const output = tokenizer.decode([result]);
        setText((state) => ({
          prompt: state.prompt,
          completion: state.completion + state.lastToken,
          lastToken: output
        }));

        if (run.current) {
          setTimeout(() => generate(false, input + output), 10);
        } else {
          setTextAreaVisibility(true);
        }
      }
    } catch (error) {
      setError(error.message);
      setTextAreaVisibility(true);
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        width: 'fit-content'
      }}
    >
      {textAreaVisible ? (
        <div>
          <TextareaAutosize
            value={text.prompt + text.completion + text.lastToken}
            onChange={(e) => {
              setText({
                prompt: e.target.value,
                completion: '',
                lastToken: ''
              });
            }}
            style={{
              width: 'calc(100vw - 26px)',
              resize: 'none',
              fontSize: '16px',
              fontFamily: 'Times',
              padding: '4px'
            }}
            minRows={10}
            maxRows={10}
            disabled={!params}
          />
        </div>
      ) : (
        <div>
          <Box
            width='calc(100vw - 16px)'
            height='200px'
            p='4px'
            border='1px solid #1D267D'
            borderRadius='3px'
            boxSizing={'border-box'}
            marginBottom={'4px'}
          >
            <pre style={{ whiteSpace: 'pre-wrap', fontSize: '16px', fontFamily: 'Times', margin: 0 }}>
              <span style={{ color: '#7EA1FF' }}>{text.prompt}</span>
              <span style={{ color: '#9CAFAA' }}>{text.completion}</span>
              <span style={{ color: '#FFFBDA', backgroundColor: '#ED9455' }}>{text.lastToken}</span>
            </pre>
          </Box>
        </div>
      )}

      {!runState && (
        <Button
          onClick={() => generate(true, text.prompt + text.completion + text.lastToken)}
          variant='outlined'
          disabled={!params}
        >
          Generate
        </Button>
      )}
      {runState && (
        <Button
          onClick={() => {
            run.current = !run.current;
            setRunState(!runState);
          }}
          variant='outlined'
        >
          Stop
        </Button>
      )}
      <TransformerVisualizer
        currentLayer={n_layer}
        n_layer={hparams.n_layer}
        topKTokens={topKTokens}
        selectedToken={text.lastToken}
      />
    </div>
  );
};

export default InferenceWindow;
