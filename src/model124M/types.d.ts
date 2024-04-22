export interface BlockParams {
  attn: {
    c_attn: { b: number[]; w: number[][] };
    c_proj: { b: number[]; w: number[][] };
  };
  ln_1: { b: number[]; g: number[] };
  ln_2: { b: number[]; g: number[] };
  mlp: {
    c_fc: { b: number[]; w: number[][] };
    c_proj: { b: number[]; w: number[][] };
  };
}

export interface Params {
  wpe: number[][];
  wte: number[][];
  ln_f: { b: number[]; g: number[] };
  blocks: BlockParams[];
}
