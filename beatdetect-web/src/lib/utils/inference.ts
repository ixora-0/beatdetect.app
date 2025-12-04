import type { Config } from '$lib/types/config';
import JSZip from 'jszip';
import { load, type NpyArray } from 'npyjs';

import { fetchBinaryCached, getModelURL } from '$lib/utils/model-registry';

// Type definitions
type State = [number, number, number]; // period idx, ts idx, phase
type Observation = [number, number, number]; // beat prob, down beat prob, no beat prob

interface Arrays {
  rowPtrs: Int32Array; // [numStates + 1]
  possibleNextStatesFlat: Int32Array; // flat transition target list [n_transitions]
  transLP: Float32Array; // transition log probs [n_transitions]
  statesFlat: Int32Array; // flatten [numStates * 3]
  statesToIdxFlat: Int32Array; // flatten 3D array
  statesToIdxShape: [number, number, number]; // [P, TS, maxPhase]
  initPeriodDist: Float32Array; // [numPeriods]
}

async function loadNPZ(buffer: ArrayBuffer | Uint8Array) {
  const zip = await JSZip.loadAsync(buffer);
  const result: Record<string, NpyArray> = {};

  for (const [filename, file] of Object.entries(zip.files)) {
    if (!filename.endsWith('.npy')) continue;

    const npyBuffer = await file.async('arraybuffer');
    const arr = await load(npyBuffer);
    result[filename.replace(/\.npy$/, '')] = arr;
  }

  return result;
}

export async function loadArrays(): Promise<Arrays> {
  const result: Partial<Arrays> = {};

  const transUrl = await getModelURL('transitions');
  const initUrl = await getModelURL('init_dist');

  const transBuf = await fetchBinaryCached(transUrl);
  const initBuf = await fetchBinaryCached(initUrl);

  for (const buffer of [transBuf, initBuf]) {
    const data = await loadNPZ(buffer);

    for (const [key, arr] of Object.entries(data)) {
      switch (key) {
        case 'row_ptrs':
          result.rowPtrs = arr.data as Int32Array;
          break;

        case 'possible_next_states':
          result.possibleNextStatesFlat = arr.data as Int32Array;
          break;

        case 'lp':
          result.transLP = arr.data as Float32Array;
          break;

        case 'states':
          result.statesFlat = arr.data as Int32Array;
          break;

        case 'states_to_idx':
          result.statesToIdxFlat = arr.data as Int32Array;
          result.statesToIdxShape = arr.shape as [number, number, number];
          break;

        case 'period':
          result.initPeriodDist = arr.data as Float32Array;
          break;
      }
    }
  }

  const missing = Object.entries(result).filter(([, v]) => v == null);
  if (missing.length > 0) {
    const names = missing.map(([k]) => k).join(', ');
    throw new Error(`loadArrays(): Missing required fields: ${names}`);
  }

  return result as Arrays;
}

class Node {
  lp: number;
  state: State;
  prev: Node | null;
  frame: number; // need to be rounded when use for indexing

  constructor(lp: number, state: State, prev: Node | null = null, frame = 0) {
    this.lp = lp;
    this.state = state;
    this.prev = prev;
    this.frame = frame;
  }
}

function posPerBeat(period: number, fps: number): number {
  const basePeriod = (60 / 120) * fps; // period for 120 BPM reference
  const basePosPerBeat = 3; // 3 positions per beat at 120 BPM
  return Math.round((basePosPerBeat * period) / basePeriod);
}

function framesPerPosition(period: number, fps: number): number {
  return period / posPerBeat(period, fps);
}

function emissionLP(isDownbeat: boolean, isBeat: boolean, observation: Observation): number {
  const [beatProb, downbeatProb, noBeatProb] = observation;
  const eps = 1e-9;
  if (isDownbeat) return Math.log(downbeatProb + eps);
  if (isBeat) return Math.log(beatProb + eps);
  return Math.log(noBeatProb + eps);
}

function defaultBeamWidth(T: number, c = 450000, wMin = 64, wMax = 1024): number {
  const w = Math.round(c / T);
  return Math.max(wMin, Math.min(wMax, w));
}

// Get observation vector at time index t
function getObservation(nnOutput: Float32Array<ArrayBufferLike>, t: number): Observation {
  // assume 3 channels
  const T = nnOutput.length / 3;
  return [nnOutput[0 * T + t], nnOutput[1 * T + t], nnOutput[2 * T + t]];
}

// Read state at index i from statesFlat
function getState(statesFlat: Int32Array, i: number): State {
  return [statesFlat[i * 3], statesFlat[i * 3 + 1], statesFlat[i * 3 + 2]];
}

function getStateIdx(
  stateToIdxFlat: Int32Array,
  stateToIdxShape: [number, number, number],
  state: State
): number {
  const [periodIdx, tsIdx, phase] = state;
  const [_P, TS, maxPhase] = stateToIdxShape;
  // index formula: ((p * TS) + ts) * maxPhase + phase
  return stateToIdxFlat[(periodIdx * TS + tsIdx) * maxPhase + phase];
}

// Backtrack returns an array of [time_sec, beat_num]
function backtrack(node: Node | null, fps: number, periods: number[]): number[][] {
  const path: Node[] = [];
  let n: Node | null = node;
  while (n !== null) {
    path.push(n);
    n = n.prev;
  }
  path.reverse();

  const beats: number[][] = [];
  let t = 0; // time in frames (but later converted to seconds incrementally)
  for (const nd of path) {
    const [periodIdx, _ts, phase] = nd.state;
    const period = periods[periodIdx];
    if (phase % posPerBeat(period, fps) === 0) {
      const beatNum = Math.floor(phase / posPerBeat(period, fps));
      // append [time_in_seconds, beat_number(1-based)]
      beats.push([t / fps, beatNum + 1]);
    }
    t += framesPerPosition(period, fps);
  }
  return beats;
}

export function beamSearch(
  config: Config,
  nnOutput: Float32Array<ArrayBufferLike>,
  arrays: Arrays,
  progressCallback?: (progress: number) => void,
  beamWidth?: number
): { beats: number[][]; bestLP: number } {
  const T = nnOutput.length / 3;
  const fps: number = config.spectrogram.fps;

  if (beamWidth === undefined || beamWidth === null) {
    beamWidth = defaultBeamWidth(T);
  }

  const periods: number[] = (config.post.tempo_bins as number[])
    .map((tempo) => (60 / tempo) * fps)
    .sort((a, b) => a - b);

  // wrapper for emissionLP
  function calcEmitLP(state: State, observation: Observation): number {
    const periodIdx = state[0];
    const phase = state[2];
    const period = periods[periodIdx];
    const isDownbeat = phase === 0;
    const isBeat = phase % posPerBeat(period, fps) === 0;
    return emissionLP(isDownbeat, isBeat, observation);
  }

  // Unpack arrays
  const rowPtr = arrays.rowPtrs;
  const possibleNextStatesFlat = arrays.possibleNextStatesFlat;
  const transLP = arrays.transLP;
  const statesFlat = arrays.statesFlat;
  const statesToIdxFlat = arrays.statesToIdxFlat;
  const statesToIdxShape = arrays.statesToIdxShape;
  const initPeriodDist = arrays.initPeriodDist;

  const numStates = statesFlat.length / 3;

  // compute start-state probabilities (like python)
  const startStatesProbs = new Float32Array(numStates);
  for (let i = 0; i < numStates; ++i) {
    const st = getState(statesFlat, i);
    const periodIdx = st[0];
    const emitLp = calcEmitLP(st, getObservation(nnOutput, 0));
    startStatesProbs[i] = initPeriodDist[periodIdx] + emitLp;
  }

  // select initial beam (top-k)
  // create an array of indices then partial sort: for simplicity use sorting (numStates ~ 10k ok)
  const idxs = Array.from({ length: numStates }, (_, i) => i);
  idxs.sort((a, b) => startStatesProbs[a] - startStatesProbs[b]); // ascending
  const startStatesIdxs = idxs.slice(Math.max(0, idxs.length - beamWidth));
  let beam: Node[] = startStatesIdxs.map(
    (i) => new Node(startStatesProbs[i], getState(statesFlat, i))
  );

  let bestNode: Node | null = null;
  let bestLP = -Infinity;
  let depth = 0;
  while (beam.length > 0) {
    const node = beam.pop()!;

    const periodIdx = node.state[0];
    const period = periods[periodIdx];
    const nextFrame = node.frame + framesPerPosition(period, fps);

    if (Math.round(nextFrame) >= T) {
      if (node.lp > bestLP) {
        bestNode = node;
        bestLP = node.lp;
      }
      continue;
    }

    depth = Math.max(nextFrame, depth);
    if (progressCallback) progressCallback((depth / T) * 100);

    const nextObservation = getObservation(nnOutput, Math.round(nextFrame));

    const stateIdx = getStateIdx(statesToIdxFlat, statesToIdxShape, node.state);
    const start = rowPtr[stateIdx];
    const end = rowPtr[stateIdx + 1];

    for (let p = start; p < end; ++p) {
      const nextStateIdx = possibleNextStatesFlat[p];

      const nextStateTuple = getState(statesFlat, nextStateIdx);
      const emitLogp = calcEmitLP(nextStateTuple, nextObservation);

      const nextLP = node.lp + transLP[p] + emitLogp;
      beam.push(new Node(nextLP, nextStateTuple, node, nextFrame));
    }

    // keep top beamWidth nodes (same behaviour as python: sort ascending, keep last beamWidth)
    beam.sort((a, b) => a.lp - b.lp);
    if (beam.length > beamWidth) {
      beam = beam.slice(beam.length - beamWidth);
    }
  }

  const beats = backtrack(bestNode, fps, periods);
  return { beats, bestLP };
}
