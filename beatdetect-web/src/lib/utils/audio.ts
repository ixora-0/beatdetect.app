export async function decodeAudio(file: File) {
  const audioContext = new window.AudioContext();
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  return audioBuffer;
}

export async function resampleBuffer(audioBuffer: AudioBuffer, sampleRate: number) {
  let resampledBuffer = audioBuffer;
  if (audioBuffer.sampleRate !== sampleRate) {
    const offlineContext = new OfflineAudioContext(
      audioBuffer.numberOfChannels,
      audioBuffer.duration * sampleRate,
      sampleRate
    );
    const source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineContext.destination);
    source.start();
    resampledBuffer = await offlineContext.startRendering();
  }
  return resampledBuffer;
}

export async function toMono(
  audioBuffer: AudioBuffer,
  progressCallback?: (progress: number) => void
) {
  // Convert to mono
  const mono = new Float32Array(audioBuffer.length);
  for (let i = 0; i < audioBuffer.length; i++) {
    let sum = 0;
    // Average channels
    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
      sum += audioBuffer.getChannelData(channel)[i];
    }
    mono[i] = sum / audioBuffer.numberOfChannels;
    if (progressCallback) progressCallback((i / audioBuffer.length) * 100);
  }
  return mono;
}
