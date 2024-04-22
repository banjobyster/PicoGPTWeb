import { matrixMultiplication } from '../common/matrixUtils';

class MatrixOperations {
  static async matrixMultiply(matrix1: number[][], matrix2: number[][]): Promise<number[][]> {
    const res = await matrixMultiplication({ matA: matrix1, matB: matrix2 });
    return res ?? [[]];
  }

  static transposeMatrix(matrix: number[][]): number[][] {
    const numRows = matrix.length;
    const numCols = matrix[0].length;
    const transposedMatrix: number[][] = [];

    for (let i = 0; i < numCols; i++) {
      transposedMatrix[i] = [];
      for (let j = 0; j < numRows; j++) {
        transposedMatrix[i][j] = matrix[j][i];
      }
    }

    return transposedMatrix;
  }
}

class ActivationFunctions {
  static gelu(x: number[][]): number[][] {
    return x.map((row) =>
      row.map(
        (val) =>
          0.5 * val * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (val + 0.044715 * Math.pow(val, 3))))
      )
    );
  }

  static softmax(x: number[][]): number[][] {
    const maxVal = x.map((row) => Math.max(...row));
    const exp_x = x.map((row, i) => row.map((val) => Math.exp(val - maxVal[i])));
    const sumExp = exp_x.map((row) => row.reduce((acc, val) => acc + val, 0));
    return exp_x.map((row, i) => row.map((val) => val / sumExp[i]));
  }
}

class LayerOperations {
  static layerNorm(x: number[][], g: number[], b: number[], eps: number = 1e-5): number[][] {
    const mean = x.map((row) => row.reduce((acc, val) => acc + val, 0) / row.length);
    const variance = x.map(
      (row, i) => row.reduce((acc, val) => acc + Math.pow(val - mean[i], 2), 0) / row.length
    );
    x = x.map((row, i) => row.map((val) => (val - mean[i]) / Math.sqrt(variance[i] + eps)));
    return x.map((row) => row.map((val, i) => g[i] * val + b[i]));
  }
}

class NeuralNetwork {
  static async linear(x: number[][], w: number[][], b: number[]): Promise<number[][]> {
    return (await MatrixOperations.matrixMultiply(x, w)).map((row) =>
      row.map((val, i) => val + b[i])
    );
  }

  static async ffn(x: number[][], c_fc: any, c_proj: any): Promise<number[][]> {
    const a = ActivationFunctions.gelu(await NeuralNetwork.linear(x, c_fc.w, c_fc.b));
    return await NeuralNetwork.linear(a, c_proj.w, c_proj.b);
  }
}

class Attention {
  static async attention(
    q: number[][],
    k: number[][],
    v: number[][],
    mask: number[][]
  ): Promise<number[][]> {
    const kT = MatrixOperations.transposeMatrix(k);
    const qkT = await MatrixOperations.matrixMultiply(q, kT);
    const sqrtQ = Math.sqrt(q[0].length);
    const qkTScaled = qkT.map((qkRow) => qkRow.map((val) => val / sqrtQ));
    const maskedQK = qkTScaled.map((qkRow, i) => qkRow.map((val, j) => val + mask[i][j]));
    const softmaxQK = ActivationFunctions.softmax(maskedQK);
    return await MatrixOperations.matrixMultiply(softmaxQK, v);
  }
}

class Splitting {
  static split(x: number[][], n: number): number[][][] {
    return Array.from({ length: n }, (_, i) =>
      x.map((row) => row.slice((i * row.length) / n, ((i + 1) * row.length) / n))
    );
  }
}

class MultiHeadAttention {
  static async mha(x: number[][], c_attn: any, c_proj: any, n_head: number): Promise<number[][]> {
    const xProjected = await NeuralNetwork.linear(x, c_attn.w, c_attn.b);
    const qkv = Splitting.split(xProjected, 3);
    const qkv_heads = qkv.map((qkvRow) => Splitting.split(qkvRow, n_head));
    const causal_mask = Array.from({ length: x.length }, (_, i) =>
      Array.from({ length: x.length }, (_, j) => (1 - (i >= j ? 1 : 0)) * -1e10)
    );
    const out_heads: number[][][] = [];
    for (let i = 0; i < n_head; i++) {
      const q = qkv_heads[0][i];
      const k = qkv_heads[1][i];
      const v = qkv_heads[2][i];
      const result = await Attention.attention(q, k, v, causal_mask);
      out_heads.push(result);
    }

    const mergedHeads = out_heads[0].map((_, idx) =>
      out_heads.map((head) => head[idx]).reduce((acc, val) => acc.concat(val), [])
    );
    return await NeuralNetwork.linear(mergedHeads, c_proj.w, c_proj.b);
  }
}

class TransformerBlock {
  static async transformer_block(
    x: number[][],
    mlp: any,
    attn: any,
    ln_1: any,
    ln_2: any,
    n_head: number
  ): Promise<number[][]> {
    const norm1 = LayerOperations.layerNorm(x, ln_1.g, ln_1.b);
    const mhaOut = await MultiHeadAttention.mha(norm1, attn.c_attn, attn.c_proj, n_head);
    const add1 = x.map((row, i) => row.map((elem, j) => elem + mhaOut[i][j]));

    const norm2 = LayerOperations.layerNorm(add1, ln_2.g, ln_2.b);
    const ffnOut = await NeuralNetwork.ffn(norm2, mlp.c_fc, mlp.c_proj);
    return add1.map((row, i) => row.map((elem, j) => elem + ffnOut[i][j]));
  }
}

class GPT2 {
  static async gpt2(
    inputs: number[],
    wte: number[][],
    wteT: number[][],
    wpe: number[][],
    blocks: any[],
    ln_f: any,
    n_head: number,
    updateNetworkState: (props) => void
  ): Promise<number[][]> {
    let x = inputs.map((input, idx) => wte[input].map((value, i) => value + wpe[idx][i]));
    let blockNumber = 0;
    for (const block of blocks) {
      x = await TransformerBlock.transformer_block(
        x,
        block.mlp,
        block.attn,
        block.ln_1,
        block.ln_2,
        n_head
      );
      updateNetworkState({ currentLayer: blockNumber });
      blockNumber += 1;
    }
    const layerNormalizedX = LayerOperations.layerNorm(x, ln_f.g, ln_f.b);
    return await MatrixOperations.matrixMultiply(layerNormalizedX, wteT);
  }

  static async generate(
    inputs: number[],
    params: any,
    n_head: number,
    updateNetworkState: (props) => void
  ): Promise<number> {
    const logits = await GPT2.gpt2(
      inputs,
      params.wte,
      params.wteT,
      params.wpe,
      params.blocks,
      params.ln_f,
      n_head,
      updateNetworkState
    );

    const topKTokens = logits[logits.length - 1]
      .map((val, idx) => [val, idx])
      .sort(([valA], [valB]) => valB - valA)
      .slice(0, 5)
      .map(([_, idx]) => idx);
    updateNetworkState({ topKTokens: topKTokens });

    const randomSamplingWithBias = (topKTokens: number[]) => {
      const temperature = 0.7;
      const logitsExp = topKTokens.map((idx) =>
        Math.exp(logits[logits.length - 1][idx] / temperature)
      );
      const sumExp = logitsExp.reduce((acc, val) => acc + val, 0);
      const probs = logitsExp.map((val) => val / sumExp);

      let acc = 0;
      const r = Math.random();
      for (let i = 0; i < probs.length; i++) {
        acc += probs[i];
        if (r <= acc) return topKTokens[i];
      }

      return topKTokens[0];
    };

    return randomSamplingWithBias(topKTokens);
  }
}

export { GPT2, MatrixOperations };
