const pptxgen = require('pptxgenjs');
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9'; // 10 x 5.625
pres.title = 'Verbatim Transcription of Overlapping Meeting Speech';
pres.author = 'Shreya Mary Kurien, Krishiv Saluja, Sampath Bageyawadi, Thummala Koushik, M Sanjay';

const INK = '14213D', WHITE = 'FFFFFF', BODY = '3D4555', MUTED = '5A6272';
const ORANGE = 'C46A1C', BLUE = '2F6DB5', CARD = 'EEF2F8', NAVY2 = '24345C', ICE = 'C9D3E6', SAND = 'F2B27A';
const H = 'Cambria', B = 'Calibri';
const M = 0.5, W = 10 - 2 * M;

function eyebrow(s, t, dark) {
  s.addText(t.toUpperCase(), { x: M, y: 0.38, w: W, h: 0.3, fontFace: B, fontSize: 11, bold: true, charSpacing: 2, color: dark ? SAND : ORANGE, margin: 0, isTextBox: true });
}
function title(s, t, dark) {
  s.addText(t, { x: M, y: 0.68, w: W, h: 0.55, fontFace: H, fontSize: 24, bold: true, color: dark ? WHITE : INK, margin: 0, valign: 'top', isTextBox: true });
}
function content(eb, t, dark) {
  const s = pres.addSlide();
  s.background = { color: dark ? INK : WHITE };
  eyebrow(s, eb, dark); title(s, t, dark);
  return s;
}
function card(s, x, y, w, h, head, body, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: opts.fill || CARD }, line: { color: opts.fill || CARD }, rectRadius: 0.08 });
  s.addText(head, { x: x + 0.2, y: y + 0.18, w: w - 0.4, h: 0.4, fontFace: B, fontSize: 15, bold: true, color: opts.headColor || INK, margin: 0, valign: 'top', isTextBox: true });
  s.addText(body, { x: x + 0.2, y: y + 0.6, w: w - 0.4, h: h - 0.75, fontFace: B, fontSize: 12.5, color: opts.bodyColor || BODY, margin: 0, valign: 'top', isTextBox: true });
}
function stat(s, x, y, w, num, label, color, size = 40) {
  s.addText(num, { x, y, w, h: 0.8, fontFace: H, fontSize: size, bold: true, color, margin: 0, valign: 'bottom', isTextBox: true });
  s.addText(label, { x, y: y + 0.88, w, h: 0.9, fontFace: B, fontSize: 13, color: BODY, margin: 0, valign: 'top', isTextBox: true });
}
function circleNum(s, x, y, n, color) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color }, line: { color } });
  s.addText(String(n), { x, y, w: 0.42, h: 0.42, fontFace: B, fontSize: 14, bold: true, color: WHITE, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
}

// 1 cover
{
  const s = pres.addSlide(); s.background = { color: INK };
  s.addText('DEEP LEARNING PROJECT-BASED LEARNING · WOXSEN UNIVERSITY', { x: M, y: 0.6, w: W, h: 0.3, fontFace: B, fontSize: 11, bold: true, charSpacing: 2, color: SAND, margin: 0, isTextBox: true });
  s.addText('Verbatim transcription of overlapping meeting speech, on a laptop', { x: M, y: 1.0, w: 8.2, h: 1.7, fontFace: H, fontSize: 36, bold: true, color: WHITE, margin: 0, valign: 'top', isTextBox: true });
  s.addText('Who spoke, what they said — including every “um” — when two people talk at once', { x: M, y: 2.8, w: 8, h: 0.6, fontFace: B, fontSize: 16, color: ICE, margin: 0, isTextBox: true });
  s.addText('Shreya Mary Kurien · Krishiv Saluja · Sampath Bageyawadi · Thummala Koushik · M Sanjay\nDepartment of Computer Science and Engineering', { x: M, y: 4.45, w: W, h: 0.65, fontFace: B, fontSize: 12, color: ICE, margin: 0, valign: 'bottom', isTextBox: true });
  s.addNotes('Introduce the team and the one-line goal: a transcript of a real room recording that says who spoke, keeps exactly what was said including fillers, and handles people talking over each other — all with open models on a MacBook Air.');
}

// 2 problem
{
  const s = content('The problem', 'One microphone, four problems at once');
  const items = [
    ['Who speaks?', 'Consistent Speaker_N labels for any set of people, with no per-speaker training.', BLUE],
    ['Overlap', 'Two people talk at once; one mixed waveform holds both voices.', ORANGE],
    ['Verbatim', 'Stock Whisper deletes “um”, “uh”, “mm-hmm”: it kept 3 of 128 fillers in our test.', BLUE],
    ['Live', 'Fast enough to follow a conversation on consumer hardware.', ORANGE],
  ];
  const cw = (W - 3 * 0.25) / 4;
  items.forEach(([h, b, c], i) => {
    const x = M + i * (cw + 0.25);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: cw, h: 2.9, fill: { color: CARD }, line: { color: CARD }, rectRadius: 0.08 });
    circleNum(s, x + 0.2, 1.95, i + 1, c);
    s.addText(h, { x: x + 0.2, y: 2.5, w: cw - 0.4, h: 0.45, fontFace: B, fontSize: 15, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(b, { x: x + 0.2, y: 3.0, w: cw - 0.4, h: 1.5, fontFace: B, fontSize: 12.5, color: BODY, margin: 0, valign: 'top', isTextBox: true });
  });
  s.addNotes('Meeting transcripts from a single shared microphone need all four at once. Off-the-shelf ASR is strong on clean single-speaker audio but deletes disfluencies and has no notion of speakers. Target users: minute takers, conversation researchers, accessibility.');
}

// 3 pipeline
{
  const s = content('System', 'Four open models, chained');
  const nodes = [
    ['Diarisation', 'pyannote 3.1 — who speaks when, overlaps kept', INK],
    ['Routing', 'single-speaker vs overlapped segments', INK],
    ['Separation', 'SepFormer splits 2 voices; ECAPA matches each to a speaker', ORANGE],
    ['Verbatim ASR', 'Whisper-small, adapted on AMI', BLUE],
  ];
  const nw = 1.95, gap = (W - 4 * nw) / 3;
  nodes.forEach(([h, b, c], i) => {
    const x = M + i * (nw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: nw, h: 1.55, fill: { color: WHITE }, line: { color: c, width: 1.75 }, rectRadius: 0.08 });
    s.addText(h, { x: x + 0.15, y: 1.88, w: nw - 0.3, h: 0.4, fontFace: B, fontSize: 14, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(b, { x: x + 0.15, y: 2.3, w: nw - 0.3, h: 0.95, fontFace: B, fontSize: 11.5, color: MUTED, margin: 0, valign: 'top', isTextBox: true });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + nw + 0.1, y: 2.4, w: gap - 0.2, h: 0.26, fill: { color: ORANGE }, line: { color: ORANGE } });
  });
  s.addText([{ text: 'Single-speaker segments skip separation. ', options: { bold: true } }, { text: 'Separated streams are matched to speakers by voice print, because separator output order is arbitrary.' }],
    { x: M, y: 3.65, w: 4.3, h: 1.0, fontFace: B, fontSize: 13, color: BODY, margin: 0, valign: 'top', isTextBox: true });
  s.addText([{ text: 'Lines merge by start time, ', options: { bold: true } }, { text: 'so interruptions show as separate, correctly ordered lines. Optional enrolment swaps Speaker_N for a real name.' }],
    { x: 5.2, y: 3.65, w: 4.3, h: 1.0, fontFace: B, fontSize: 13, color: BODY, margin: 0, valign: 'top', isTextBox: true });
  s.addNotes('Walk through the chain. One diarisation pass gives both speech boundaries and overlap flags, so no separate VAD. Only overlapped segments pay for separation. The Hungarian assignment of streams to speakers uses ECAPA voice prints from each speaker\'s clean speech.');
}

// 4 overlap
{
  const s = content('Overlap', 'Separation recovers the hidden speaker');
  const cw = (W - 2 * 0.4) / 3;
  stat(s, M, 1.7, cw, '19.0 dB', 'SI-SDR improvement, two synthetic voices (Libri2Mix SepFormer; 7.6 dB WSJ0-2mix, 1.5 dB WHAMR!)', ORANGE, 34);
  stat(s, M + cw + 0.4, 1.7, cw, '12.4 dB', 'SI-SDR improvement, real recorded voice + synthetic voice, 6.5 s overlap', BLUE, 34);
  stat(s, M + 2 * (cw + 0.4), 1.7, cw, '45% → 0%', 'cpWER on the synthetic two-speaker demo, without vs with separation', ORANGE, 34);
  s.addText('Single mixtures each, clean digital sums — indicative, not a benchmark. Overlap detection placed the overlap at 5.01–7.56 s vs a true 5.0–7.5 s.',
    { x: M, y: 4.6, w: W, h: 0.5, fontFace: B, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true });
  s.addNotes('Three public SepFormer checkpoints compared. The clean-trained Libri2Mix model won clearly, but the test mixtures are clean digital sums, which favours it. Without separation the overlapped stretch collapsed into one mixed line (45% cpWER); with it both speakers came out correctly.');
}

// 5 speed
{
  const s = content('Live mode', 'Measuring cost made it near-real-time');
  stat(s, M, 1.6, 4.2, '5.2×', 'faster Whisper on the Apple GPU (MLX) vs CPU, identical transcript: 0.45 s vs 2.34 s for 4.4 s of audio', ORANGE, 44);
  stat(s, 5.2, 1.6, 4.3, '14.5 → 8 s', 'mean speech-to-line latency on a two-speaker clip; the maximum fell from 25 s and stopped growing', BLUE, 44);
  card(s, M, 3.65, 4.3, 1.4, 'Call cost is fixed', 'Whisper always encodes a 30 s window, so cost scales with the number of calls: clean clips are packed into one call.');
  card(s, 5.2, 3.65, 4.3, 1.4, 'But not overlaps', 'Packing separated streams lost a whole utterance, so those are transcribed alone.');
  s.addNotes('Rolling 16 s window re-diarised every 4 s, 3 s guard so lines are never retracted. The gains came from measuring where time goes: GPU Whisper, GPU diarisation, packing clean clips. Packing everything was faster but lost words in overlaps, so accuracy won. Latency is roughly 7-8 s.');
}

// 6 adapt
{
  const s = content('Verbatim adaptation', 'Fine-tuning Whisper on a 16 GB fanless laptop');
  const cw = (W - 2 * 0.25) / 3;
  card(s, M, 1.7, cw, 2.9, 'Data', 'AMI meeting corpus, single distant mic, verbatim transcripts with fillers (CC BY 4.0).\n\n17,704 windows · 21.7 h · 78 meetings.');
  card(s, M + cw + 0.25, 1.7, cw, 2.9, 'Cached encoder', 'Frozen lower 10 encoder layers run once; the top 2 layers + decoder train from the cache.\n\nThe encoder dominated compute (2.8 s vs 0.46 s).');
  card(s, M + 2 * (cw + 0.25), 1.7, cw, 2.9, 'Recipe', 'AdamW 1e-5, weight decay 0.01, latent masking, batch 2 × 4.\n\nBest validation WER at step 200 of 1,400 — that checkpoint is used.');
  s.addNotes('Full fine-tuning on the M3 would take hours per thousand steps. Caching the frozen lower layers makes it feasible. Micro-batch 4 swapped the 16 GB machine, so 2 with accumulation 4. Validation WER peaked at the first check (step 200) and never improved, so the short adaptation does the work; a lower learning rate or shorter schedule is the next experiment.');
}

// 7 mismatch
{
  const s = content('What went wrong first', 'The first model learned to transcribe everyone');
  stat(s, M, 1.6, 4.2, '+3.7 pts', 'WER of the first fine-tune vs stock (38.8% → 42.5%), despite keeping 74% of fillers', ORANGE, 44);
  stat(s, 5.2, 1.6, 4.3, '10% → 29%', 'insertions as a share of reference words: deletions halved, insertions nearly tripled', BLUE, 44);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 3.7, w: W, h: 1.3, fill: { color: CARD }, line: { color: CARD }, rectRadius: 0.08 });
  s.addText([
    { text: 'Training windows held ' }, { text: 'every speaker’s words', options: { bold: true } },
    { text: '. Inside the pipeline the recogniser only ever hears ' }, { text: 'one speaker', options: { bold: true } },
    { text: ' — a clean segment or a separated stream. The model transcribed whoever was audible, and those words counted as errors.' }],
    { x: M + 0.25, y: 3.8, w: W - 0.5, h: 1.1, fontFace: B, fontSize: 14, color: INK, margin: 0, valign: 'middle', isTextBox: true });
  s.addNotes('The key lesson of the project. The first adaptation fixed fillers but made WER worse. The error breakdown pointed at insertions; inspection showed the cause: a training target that does not match what the model sees at inference.');
}

// 8 fix (dark)
{
  const s = content('The fix', 'Train on exactly what the pipeline will feed it', true);
  const cw = (W - 2 * 0.25) / 3;
  const f = { fill: NAVY2, headColor: WHITE, bodyColor: ICE };
  card(s, M, 1.75, cw, 2.7, 'Same routing code', 'The pipeline’s own overlap-routing function picks single-speaker stretches; only those become training windows.', f);
  card(s, M + cw + 0.25, 1.75, cw, 2.7, 'One speaker each', 'Every example is “this audio, this one speaker’s words”. Overlaps and repetition-loop windows are excluded.', f);
  card(s, M + 2 * (cw + 0.25), 1.75, cw, 2.7, 'Honest selection', 'Weight decay, latent masking, and the checkpoint chosen by validation WER, not the last step.', f);
  s.addNotes('Same model size, layers, data source and budget. The change is the training data specification, made exact by reusing the inference routing code to build it, plus standard regularisation and checkpoint selection on validation WER.');
}

// 9 results (chart)
{
  const s = content('Result · 300 held-out AMI utterances', 'Fillers kept, with no loss in accuracy');
  s.addChart(pres.charts.BAR, [
    { name: 'Stock Whisper-small', labels: ['WER', 'Verbatim WER', 'Filler recall'], values: [33.9, 37.7, 2.3] },
    { name: 'Adapted', labels: ['WER', 'Verbatim WER', 'Filler recall'], values: [31.5, 32.2, 64.8] },
  ], {
    x: M, y: 1.5, w: 5.6, h: 3.6, barDir: 'col', barGapWidthPct: 60,
    chartColors: ['9AA5B8', ORANGE], showLegend: true, legendPos: 'b', legendFontSize: 11, legendFontFace: B, legendColor: BODY,
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFontSize: 11, dataLabelFontFace: B, dataLabelColor: INK, dataLabelFormatCode: '0.0',
    catAxisLabelColor: BODY, catAxisLabelFontFace: B, catAxisLabelFontSize: 12, valAxisLabelColor: MUTED, valAxisLabelFontSize: 10,
    valAxisMaxVal: 80, valAxisMinVal: 0, valGridLine: { color: 'E3E6EC', size: 0.5 }, catGridLine: { style: 'none' },
    showTitle: true, title: 'Percent (lower is better for WER, higher for filler recall)', titleFontSize: 11, titleColor: MUTED, titleFontFace: B,
  });
  stat(s, 6.4, 1.6, 3.1, '2% → 65%', 'filler recall: 83 of 128 reference fillers kept, vs 3', ORANGE, 34);
  stat(s, 6.4, 3.25, 3.1, '37.7 → 32.2', 'verbatim WER (%), fillers counted as words', BLUE, 34);
  s.addText('95% bootstrap intervals: filler recall 0–5% vs 53–75%; verbatim WER 34–42% vs 28–36%; WER 30–38% vs 28–36% (overlapping).',
    { x: 6.4, y: 4.75, w: 3.1, h: 0.6, fontFace: B, fontSize: 9.5, italic: true, color: MUTED, margin: 0, valign: 'top', isTextBox: true });
  s.addNotes('Same 300 utterances from 16 held-out meetings for both models. Filler recall and verbatim WER improve by margins far larger than their intervals. Plain WER is lower too, but the two intervals overlap, so we only claim it is no worse; a paired test on the same utterances is the right check and our scripts provide it.');
}

// 10 meetings (table)
{
  const s = content('Full pipeline', 'Three 4-person AMI meetings, end to end');
  const hdr = { bold: true, color: WHITE, fill: { color: INK }, fontFace: B, fontSize: 12.5 };
  const c = { color: BODY, fontFace: B, fontSize: 12.5 };
  const tot = { bold: true, color: INK, fill: { color: CARD }, fontFace: B, fontSize: 12.5 };
  const rows = [
    ['Meeting', 'DER', 'cpWER stock', 'cpWER adapted', 'Fillers kept'].map(t => ({ text: t, options: hdr })),
    ...[['ES2004c', '9.4%', '58.7%', '61.9%', '1 → 14 of 14'], ['IS1009b', '25.4%', '56.0%', '53.8%', '1 → 30 of 39'], ['EN2002b', '23.5%', '55.0%', '56.3%', '0 → 21 of 30']]
      .map(r => r.map(t => ({ text: t, options: c }))),
    ['Mean / total', '19.4%', '56.6%', '57.3%', '2 → 65 of 83'].map(t => ({ text: t, options: tot })),
  ];
  s.addTable(rows, { x: M, y: 1.6, w: W, colW: [2.0, 1.2, 1.9, 1.9, 2.0], rowH: 0.42, border: { type: 'solid', pt: 0.5, color: 'D5DAE3' }, valign: 'middle' });
  s.addText([{ text: 'Fillers: 2% → 78%. ', options: { bold: true, color: INK } }, { text: 'Word accuracy moves −2 to +3 points, inside intervals tens of points wide: three 5-minute excerpts cannot separate the two recognisers.' }],
    { x: M, y: 4.0, w: W, h: 0.9, fontFace: B, fontSize: 14, color: BODY, margin: 0, valign: 'top', isTextBox: true });
  s.addNotes('cpWER penalises wrong words and words credited to the wrong person. DER is the same for both rows because the diariser is the same. The clear meeting-level effect is filler retention. Three excerpts are too few for a word-accuracy verdict.');
}

// 11 errors
{
  const s = content('Where errors come from', 'Diarisation, not recognition, dominates');
  const cw = (W - 2 * 0.4) / 3;
  stat(s, M, 1.7, cw, '21%', 'speaker confusion in IS1009b: two voices labelled consistently only about half the time', ORANGE, 44);
  stat(s, M + cw + 0.4, 1.7, cw, '15%', 'of speech missed in EN2002b, where 41% of speech overlaps and 24 s has 3+ talkers', BLUE, 44);
  stat(s, M + 2 * (cw + 0.4), 1.7, cw, '78 s', 'of ES2004c speech has no reference text, so correct words there score as insertions', ORANGE, 44);
  s.addText('Every diarisation error propagates to cpWER, and the separator recovers at most two simultaneous voices by design.',
    { x: M, y: 4.3, w: W, h: 0.6, fontFace: B, fontSize: 14, color: BODY, margin: 0, isTextBox: true });
  s.addNotes('Why meeting cpWER (~57%) is much higher than utterance WER (~32%): diarisation errors and reference gaps, not the recogniser. An earlier evaluation built the DER reference from text turns and reported a spurious 57% DER; with complete annotation it is 9.4%.');
}

// 12 limits
{
  const s = content('Limitations', 'What we can and cannot claim');
  const items = [
    ['Small evaluation', '300 utterances; three 5-minute meetings; separation measured on single synthetic mixtures.'],
    ['Two-source separation', 'Three or more simultaneous speakers cannot all be recovered.'],
    ['Readability', 'The adapted model writes lower-case, unpunctuated text; punctuation restoration is not built.'],
    ['Live mode', 'About 7–8 s lag; tested with replayed audio, not yet a live microphone.'],
  ];
  const cw = (W - 0.3) / 2, ch = 1.45;
  items.forEach(([h, b], i) => {
    const x = M + (i % 2) * (cw + 0.3), y = 1.65 + Math.floor(i / 2) * (ch + 0.25);
    card(s, x, y, cw, ch, h, b);
  });
  s.addNotes('We report intervals and limits explicitly. This is a systems study with indicative measurements, not a benchmark. One training configuration, no learning-rate sweep.');
}

// 13 conclusion (dark)
{
  const s = content('Conclusion', 'Match the training data to the pipeline', true);
  s.addText('An open, laptop-scale pipeline that separates overlapping speech, labels speakers, streams live, and keeps two thirds of the fillers that stock Whisper deletes — with no loss in word accuracy.',
    { x: M, y: 1.65, w: W, h: 1.5, fontFace: H, fontSize: 20, color: WHITE, margin: 0, valign: 'top', isTextBox: true });
  s.addText('NEXT', { x: M, y: 3.55, w: W, h: 0.3, fontFace: B, fontSize: 11, bold: true, charSpacing: 2, color: SAND, margin: 0, isTextBox: true });
  s.addText('Stronger diarisation · more evaluation meetings (NOTSOFAR-1, CHiME-6) · diarisation-conditioned ASR beyond two speakers · punctuation',
    { x: M, y: 3.9, w: W, h: 0.8, fontFace: B, fontSize: 15, color: ICE, margin: 0, valign: 'top', isTextBox: true });
  s.addNotes('Close on the lesson: the decisive change was not a bigger model but a training set that matches what the model hears inside the pipeline. Then the roadmap.');
}

// 14 questions
{
  const s = pres.addSlide(); s.background = { color: INK };
  s.addText('Questions?', { x: M, y: 1.6, w: W, h: 1.1, fontFace: H, fontSize: 48, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText([{ text: 'Live demo: ' }, { text: './run.sh demo', options: { bold: true, fontFace: 'Courier New' } },
    { text: ' — an AMI meeting played in real time, stock vs adapted Whisper side by side, with the ground-truth transcript.' }],
    { x: M, y: 2.9, w: 8.5, h: 1.0, fontFace: B, fontSize: 16, color: ICE, margin: 0, valign: 'top', isTextBox: true });
  s.addNotes('Offer the demo. Good examples: the synthetic tts_demo with the adapted model (um kept, interruption separated), then an AMI excerpt as the honest hard case. Caveats: lower-case unpunctuated output, 6-10 s lag, two simultaneous speakers at most.');
}

pres.writeFile({ fileName: 'presentation.pptx' }).then(f => console.log('wrote', f));
