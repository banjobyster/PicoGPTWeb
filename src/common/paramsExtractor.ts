import { MatrixOperations } from '../model124M/generate';
import hparams from '../model124M/hparams';
import { getFromIndexedDB, loadIntoIndexedDB } from './browserIndexedDB';

const downloadWeightsData = async (link: string, onProgress: any) => {
  const response = await fetch(link);
  if (!response.ok) {
    throw new Error(
      'Failed to fetch file with status (' + response.status + ') ' + response.statusText
    );
  }

  const contentLength = Number(response.headers.get('Content-Length'));
  let receivedBytes = 0;

  const reader = response.body!.getReader();

  const chunks = [];
  let MAX_ITER = 5000000000; // 500M iterations
  while (true && --MAX_ITER > 0) {
    const { done, value } = await reader.read();

    if (done) {
      break;
    }

    chunks.push(value);
    receivedBytes += value.length;

    if (contentLength) {
      const progress = Math.round((receivedBytes / contentLength) * 100);
      onProgress(progress);
    }
  }

  const buffer = new Uint8Array(receivedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  const values = new Float32Array(buffer.buffer);

  return values;
};

// The structure is inferred from the model's .ckpt.index file and hardcoded
// (124M file already takes upto 10mins to download, any bigger would be frustratingly slow)
// TODO: Find a way to infer the structure from the .index file
const extractParams = (values: Float32Array) => {
  const params = {
    wpe: [],
    wte: [],
    wteT: [],
    ln_f: { b: [], g: [] },
    blocks: Array.from({ length: 12 }, () => 0)
  };

  let index = 0;

  const blockIndices = [0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9];

  for (let i = 0; i < hparams.n_layer; i++) {
    const block = {
      attn: {
        c_attn: { b: [], w: [] },
        c_proj: { b: [], w: [] }
      },
      ln_1: { b: [], g: [] },
      ln_2: { b: [], g: [] },
      mlp: {
        c_fc: { b: [], w: [] },
        c_proj: { b: [], w: [] }
      }
    };

    block.attn.c_attn.b = values.slice(index, index + 3 * hparams.n_embd);
    index += 3 * hparams.n_embd;

    block.attn.c_attn.w = [];
    for (let j = 0; j < hparams.n_embd; j++) {
      block.attn.c_attn.w.push(values.slice(index, index + 3 * hparams.n_embd));
      index += 3 * hparams.n_embd;
    }

    block.attn.c_proj.b = values.slice(index, index + hparams.n_embd);
    index += hparams.n_embd;

    block.attn.c_proj.w = [];
    for (let j = 0; j < hparams.n_embd; j++) {
      block.attn.c_proj.w.push(values.slice(index, index + hparams.n_embd));
      index += hparams.n_embd;
    }

    block.ln_1.b = values.slice(index, index + hparams.n_embd);
    index += hparams.n_embd;

    block.ln_1.g = values.slice(index, index + hparams.n_embd);
    index += hparams.n_embd;

    block.ln_2.b = values.slice(index, index + hparams.n_embd);
    index += hparams.n_embd;

    block.ln_2.g = values.slice(index, index + hparams.n_embd);
    index += hparams.n_embd;

    block.mlp.c_fc.b = values.slice(index, index + 4 * hparams.n_embd);
    index += 4 * hparams.n_embd;

    block.mlp.c_fc.w = [];
    for (let j = 0; j < hparams.n_embd; j++) {
      block.mlp.c_fc.w.push(values.slice(index, index + 4 * hparams.n_embd));
      index += 4 * hparams.n_embd;
    }

    block.mlp.c_proj.b = values.slice(index, index + hparams.n_embd);
    index += hparams.n_embd;

    block.mlp.c_proj.w = [];
    for (let j = 0; j < 4 * hparams.n_embd; j++) {
      block.mlp.c_proj.w.push(values.slice(index, index + hparams.n_embd));
      index += hparams.n_embd;
    }

    params.blocks[blockIndices[i]] = block;
  }

  params.ln_f.b = values.slice(index, index + hparams.n_embd);
  index += hparams.n_embd;

  params.ln_f.g = values.slice(index, index + hparams.n_embd);
  index += hparams.n_embd;

  params.wpe = [];
  for (let i = 0; i < hparams.n_ctx; i++) {
    params.wpe.push(values.slice(index, index + hparams.n_embd));
    index += hparams.n_embd;
  }

  params.wte = [];
  for (let i = 0; i < hparams.n_vocab; i++) {
    params.wte.push(values.slice(index, index + hparams.n_embd));
    index += hparams.n_embd;
  }

  params.wteT = MatrixOperations.transposeMatrix(params.wte);

  return params;
};

const loadParamsFromGPT2 = async (setDownloadProgress: Dispatch<SetStateAction<number>>) => {
  try {
    const storedValues = await getFromIndexedDB();
    return extractParams(storedValues);
  } catch (error) {
    console.error('Error:', error);
  }

  const link =
    'https://huggingface.co/gpt2124mmodel/GPT2-Model-124M-For-Frontend/resolve/main/values.bin';
  const values = await downloadWeightsData(link, (progress: number) => {
    setDownloadProgress(progress);
  });

  await loadIntoIndexedDB(values);

  return extractParams(values);
};

export { loadParamsFromGPT2 };
