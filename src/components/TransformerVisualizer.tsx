import { Chip } from '@mui/material';

const TransformerBlock = ({ layer, currentLayer }: { layer: number; currentLayer: boolean }) => {
  return (
    <div
      style={{
        position: 'absolute',
        top: `calc(20px + ${layer * 5}px)`,
        left: `calc(20px + ${layer * 5}px)`,
        width: '200px',
        height: '200px',
        border: `2px solid ${currentLayer ? '#836FFF' : '#FF6868'}`,
        boxSizing: 'border-box',
        backgroundColor: `${currentLayer ? '#15F5BA' : '#DCFFB7'}`,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        color: '#0C134F',
        fontWeight: 'bold'
      }}
    >
      Transformer Block
    </div>
  );
};

const TopKTokensBlock = ({
  tokens,
  selectedToken
}: {
  tokens: string[];
  selectedToken: string;
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        margin: '10px',
        minWidth: '100px',
        color: '#0C134F',
        fontWeight: 'bold'
      }}
    >
      Top K Tokens
      {tokens.map((token, i) => {
        return (
          <Chip label={token} key={i} variant={token === selectedToken ? 'filled' : 'outlined'} />
        );
      })}
    </div>
  );
};

const TransformerVisualizer = ({
  currentLayer,
  n_layer,
  topKTokens,
  selectedToken
}: {
  currentLayer: number;
  n_layer: number;
  topKTokens: string[];
  selectedToken: string;
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'space-evenly',
        alignItems: 'center',
        width: '100%',
        marginTop: '10px',
        border: '2px solid #0C134F',
        borderRadius: '4px',
        boxSizing: 'border-box',
        padding: '4px'
      }}
    >
      <div
        style={{
          position: 'relative',
          width: '300px',
          height: '300px'
        }}
      >
        {Array.from({ length: n_layer }, (_, i) => (
          <TransformerBlock key={i} layer={i} currentLayer={i == currentLayer} />
        ))}
      </div>
      <TopKTokensBlock tokens={topKTokens} selectedToken={selectedToken} />
    </div>
  );
};

export default TransformerVisualizer;
