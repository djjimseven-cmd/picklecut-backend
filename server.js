import http from 'node:http';
import crypto from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { createWriteStream, promises as fs } from 'node:fs';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { v2 as cloudinary } from 'cloudinary';
import OpenAI from 'openai';

const PORT = Number(process.env.PORT || 10000);
const PUBLIC_BASE_URL = process.env.PUBLIC_BASE_URL || `http://localhost:${PORT}`;
const EXPORT_DIR = process.env.EXPORT_DIR || path.join(os.tmpdir(), 'picklecut-exports');
const WORK_DIR = process.env.WORK_DIR || path.join(os.tmpdir(), 'picklecut-work');

const CLOUDINARY_CLOUD_NAME = process.env.CLOUDINARY_CLOUD_NAME;
const CLOUDINARY_API_KEY = process.env.CLOUDINARY_API_KEY;
const CLOUDINARY_API_SECRET = process.env.CLOUDINARY_API_SECRET;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o';

if (CLOUDINARY_CLOUD_NAME && CLOUDINARY_API_KEY && CLOUDINARY_API_SECRET) {
  cloudinary.config({
    cloud_name: CLOUDINARY_CLOUD_NAME,
    api_key: CLOUDINARY_API_KEY,
    api_secret: CLOUDINARY_API_SECRET,
  });
}

const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

function sendJson(res, status, body) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  });
  res.end(JSON.stringify(body));
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 3_000_000) {
        reject(new Error('Request body too large'));
        req.destroy();
      }
    });
    req.on('end', () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        reject(new Error('Invalid JSON body'));
      }
    });
    req.on('error', reject);
  });
}

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve(stdout);
      else reject(new Error(`${command} exited ${code}: ${stderr.slice(-2500)}`));
    });
  });
}

async function downloadFile(url, outputPath) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Cannot download source video: ${response.status} ${response.statusText}`);
  }

  const file = createWriteStream(outputPath);

  await new Promise((resolve, reject) => {
    response.body
      .pipeTo(new WritableStream({
        write(chunk) {
          file.write(Buffer.from(chunk));
        },
        close() {
          file.end(resolve);
        },
        abort(err) {
          file.destroy();
          reject(err);
        },
      }))
      .catch(reject);
  });
}

async function getVideoDuration(videoPath) {
  const stdout = await run('ffprobe', [
    '-v', 'error',
    '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1',
    videoPath,
  ]);

  const duration = Number(stdout.trim());
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error('Cannot read video duration with ffprobe.');
  }

  return duration;
}

async function extractFrame(videoPath, timestamp, outputPath, width = 640) {
  await run('ffmpeg', [
    '-y',
    '-ss', String(Math.max(0, timestamp)),
    '-i', videoPath,
    '-frames:v', '1',
    '-vf', `scale=${width}:-1`,
    '-q:v', '3',
    outputPath,
  ]);
}

async function extractFrames(videoPath, timestamps, folder, prefix, width = 640) {
  await fs.mkdir(folder, { recursive: true });

  const frames = [];
  for (let i = 0; i < timestamps.length; i++) {
    const t = Number(timestamps[i].toFixed(2));
    const filePath = path.join(folder, `${prefix}-${String(i + 1).padStart(3, '0')}.jpg`);
    await extractFrame(videoPath, t, filePath, width);
    frames.push({ timestamp: t, filePath });
  }

  return frames;
}

async function imageToDataUrl(filePath) {
  const data = await fs.readFile(filePath);
  return `data:image/jpeg;base64,${data.toString('base64')}`;
}

function parseJsonFromText(text) {
  const cleaned = text
    .replace(/^```json/i, '')
    .replace(/^```/i, '')
    .replace(/```$/i, '')
    .trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    const first = cleaned.indexOf('{');
    const last = cleaned.lastIndexOf('}');
    if (first >= 0 && last > first) {
      return JSON.parse(cleaned.slice(first, last + 1));
    }
    throw new Error(`AI did not return valid JSON: ${text.slice(0, 500)}`);
  }
}

async function callVisionJson({ prompt, frames, maxImages = 20 }) {
  if (!openai) {
    throw new Error('Missing OPENAI_API_KEY.');
  }

  const selectedFrames = frames.slice(0, maxImages);
  const content = [{ type: 'text', text: prompt }];

  for (const frame of selectedFrames) {
    content.push({
      type: 'text',
      text: `Frame timestamp: ${frame.timestamp}s`,
    });
    content.push({
      type: 'image_url',
      image_url: {
        url: await imageToDataUrl(frame.filePath),
        detail: 'low',
      },
    });
  }

  const response = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    temperature: 0,
    messages: [
      {
        role: 'user',
        content,
      },
    ],
  });

  return parseJsonFromText(response.choices[0]?.message?.content || '{}');
}

function makeRange(start, end, step) {
  const values = [];
  for (let t = start; t <= end; t += step) {
    values.push(Number(t.toFixed(2)));
  }
  return values;
}

function dedupeTimes(times, minGap = 4) {
  const sorted = [...times]
    .map(Number)
    .filter((n) => Number.isFinite(n) && n > 0)
    .sort((a, b) => a - b);

  const result = [];
  for (const t of sorted) {
    if (!result.length || Math.abs(t - result[result.length - 1]) >= minGap) {
      result.push(t);
    }
  }
  return result;
}

function scoreCallBeforeServe(state) {
  return `${state.scoreA}-${state.scoreB}-${state.serverNumber}`;
}

function expectedSideForServingScore(score) {
  return score % 2 === 0 ? 'right/even' : 'left/odd';
}

function advanceScoreState(state, rallyWinnerTeam) {
  if (!['A', 'B'].includes(rallyWinnerTeam)) return state;

  if (rallyWinnerTeam === state.servingTeam) {
    return {
      ...state,
      scoreA: state.servingTeam === 'A' ? state.scoreA + 1 : state.scoreA,
      scoreB: state.servingTeam === 'B' ? state.scoreB + 1 : state.scoreB,
    };
  }

  if (state.serverNumber === 1) {
    return {
      ...state,
      serverNumber: 2,
      serverPlayer: state.servingTeam === 'A'
        ? (state.serverPlayer === 'A1' ? 'A2' : 'A1')
        : (state.serverPlayer === 'B1' ? 'B2' : 'B1'),
    };
  }

  const nextTeam = state.servingTeam === 'A' ? 'B' : 'A';
  return {
    ...state,
    servingTeam: nextTeam,
    serverNumber: 1,
    serverPlayer: nextTeam === 'A' ? 'A1' : 'B1',
  };
}

async function detectServeCandidates(videoPath, duration, jobDir) {
  const framesDir = path.join(jobDir, 'coarse');
  const timestamps = makeRange(0, duration, 3);
  const frames = await extractFrames(videoPath, timestamps, framesDir, 'coarse', 480);

  const candidates = [];

  for (let i = 0; i < frames.length; i += 18) {
    const batch = frames.slice(i, i + 18);

    const result = await callVisionJson({
      maxImages: 18,
      frames: batch,
      prompt: `
You are detecting pickleball doubles serve moments from sampled video frames.

Return JSON only:
{
  "serve_candidates": [
    {
      "serve_contact_seconds": number,
      "confidence": number,
      "reason": string
    }
  ]
}

Detect moments where a real point likely starts:
- players are in serve/receive formation
- server is near baseline
- receiver is diagonally opposite
- rally begins after serve

Be generous. If a serve is likely between two frames, estimate the timestamp.
Do not include warmups, ball pickup, walking, or mid-rally hits.
`,
    });

    for (const item of result.serve_candidates || []) {
      if (item.confidence >= 0.35) {
        candidates.push(item.serve_contact_seconds);
      }
    }
  }

  return dedupeTimes(candidates, 5);
}

async function validateServe(videoPath, candidateTime, matchContext, jobDir, index) {
  const start = Math.max(0, candidateTime - 2.5);
  const end = candidateTime + 2.5;
  const timestamps = makeRange(start, end, 0.5);
  const frames = await extractFrames(videoPath, timestamps, path.join(jobDir, `serve-${index}`), 'serve', 720);

  const result = await callVisionJson({
    maxImages: 12,
    frames,
    prompt: `
You are validating a pickleball doubles point start.

Match:
Team A: ${matchContext.teamA.name}
Team B: ${matchContext.teamB.name}
First serving team: ${matchContext.firstServingTeam}

Find the actual serve contact timestamp in these frames.

Point start rules:
- clip starts 1.0 second before legal serve contact
- server behind baseline
- server on correct score side if score is known
- serving partner near/behind baseline, not kitchen
- receiver behind diagonal opposite baseline
- receiver partner near kitchen/NVZ
- serve travels diagonally

Return JSON only:
{
  "serve_found": boolean,
  "serve_contact_seconds": number,
  "server_player": "A1" | "A2" | "B1" | "B2" | "unknown",
  "confidence": number,
  "player_positions": {
    "server_behind_baseline": boolean,
    "server_on_correct_score_side": boolean,
    "serving_partner_near_baseline": boolean,
    "receiver_behind_diagonal_baseline": boolean,
    "receiver_partner_near_kitchen": boolean,
    "diagonal_serve_detected": boolean
  },
  "notes": string
}

Important:
- If you see a likely real serve but cannot identify exact player, still return serve_found true and server_player unknown.
- Do not reject only because score side cannot be confirmed.
`,
  });

  return result;
}

async function detectRallyEnd(videoPath, serveTime, nextServeTime, duration, jobDir, index) {
  const searchEnd = Math.min(duration, nextServeTime ? nextServeTime - 0.5 : serveTime + 35);
  const start = serveTime + 1;
  const timestamps = makeRange(start, searchEnd, 1);
  const sampled = timestamps.slice(0, 32);
  const frames = await extractFrames(videoPath, sampled, path.join(jobDir, `end-${index}`), 'end', 720);

  const result = await callVisionJson({
    maxImages: 32,
    frames,
    prompt: `
You are detecting when a pickleball rally becomes dead.

A serve happened at ${serveTime}s.

Return JSON only:
{
  "dead_ball_found": boolean,
  "dead_ball_seconds": number,
  "dead_ball_type": "out_of_bounds" | "net_not_over" | "double_bounce" | "fault" | "hand_pickup" | "unknown",
  "rally_winner_team": "A" | "B" | "unknown",
  "confidence": number,
  "notes": string
}

End point at the earliest clear dead-ball event:
- ball lands out
- ball hits net and does not cross
- double bounce
- clear fault
- player picks up/touches dead ball by hand

If exact dead ball is unclear but players stop and someone picks up ball, use hand_pickup.
If no clear end is visible, return dead_ball_found false.
`,
  });

  if (result.dead_ball_found && Number.isFinite(Number(result.dead_ball_seconds))) {
    return result;
  }

  if (nextServeTime) {
    return {
      dead_ball_found: false,
      dead_ball_seconds: Math.max(serveTime + 3, nextServeTime - 1),
      dead_ball_type: 'unknown',
      rally_winner_team: 'unknown',
      confidence: 0.45,
      notes: 'Fallback: end set before next detected serve.',
    };
  }

  return {
    dead_ball_found: false,
    dead_ball_seconds: Math.min(duration, serveTime + 18),
    dead_ball_type: 'unknown',
    rally_winner_team: 'unknown',
    confidence: 0.35,
    notes: 'Fallback: default rally duration.',
  };
}

async function createAutocut(payload) {
  if (!payload.sourceVideoUrl) throw new Error('Missing sourceVideoUrl.');

  const jobId = randomUUID();
  const jobDir = path.join(WORK_DIR, jobId);
  await fs.mkdir(jobDir, { recursive: true });

  const sourcePath = path.join(jobDir, 'source.mp4');
  await downloadFile(payload.sourceVideoUrl, sourcePath);

  const duration = await getVideoDuration(sourcePath);

  const matchContext = {
    teamA: payload.teamA || { name: 'Team A', players: {} },
    teamB: payload.teamB || { name: 'Team B', players: {} },
    firstServingTeam: payload.firstServingTeam || 'A',
  };

  const candidates = await detectServeCandidates(sourcePath, duration, jobDir);
  const validatedServes = [];

  for (let i = 0; i < candidates.length; i++) {
    const serve = await validateServe(sourcePath, candidates[i], matchContext, jobDir, i + 1).catch(() => null);
    if (serve?.serve_found && Number(serve.serve_contact_seconds) > 0 && Number(serve.confidence || 0) >= 0.3) {
      validatedServes.push(serve);
    }
  }

  validatedServes.sort((a, b) => Number(a.serve_contact_seconds) - Number(b.serve_contact_seconds));

  const points = [];
  let state = {
    scoreA: 0,
    scoreB: 0,
    servingTeam: payload.firstServingTeam || 'A',
    serverNumber: 2,
    serverPlayer: payload.firstServingTeam === 'B' ? 'B1' : 'A1',
  };

  for (let i = 0; i < validatedServes.length; i++) {
    const serve = validatedServes[i];
    const serveTime = Number(serve.serve_contact_seconds);
    const nextServe = validatedServes[i + 1];
    const nextServeTime = nextServe ? Number(nextServe.serve_contact_seconds) : null;

    const end = await detectRallyEnd(sourcePath, serveTime, nextServeTime, duration, jobDir, i + 1);

    const pos = serve.player_positions || {};
    const startSeconds = Math.max(0, serveTime - 1);
    const endSeconds = Math.min(duration, Number(end.dead_ball_seconds) + 0.2);
    const confidence = Math.min(Number(serve.confidence || 0.5), Number(end.confidence || 0.4));

    const setupValid =
      pos.server_behind_baseline === true &&
      pos.serving_partner_near_baseline === true &&
      pos.receiver_behind_diagonal_baseline === true &&
      pos.receiver_partner_near_kitchen === true &&
      pos.diagonal_serve_detected === true;

    const servingScore = state.servingTeam === 'A' ? state.scoreA : state.scoreB;
    const scoreBefore = scoreCallBeforeServe(state);

    points.push({
      label: `Point ${i + 1}`,
      order: i + 1,
      start_seconds: startSeconds,
      serve_contact_seconds: serveTime,
      end_seconds: endSeconds,
      dead_ball_seconds: Number(end.dead_ball_seconds),
      setup_valid: setupValid,
      serve_legal: setupValid,
      end_valid: end.dead_ball_found === true,
      score_call_before_serve: scoreBefore,
      expected_server_side: expectedSideForServingScore(servingScore),
      server_player: serve.server_player || state.serverPlayer || 'unknown',
      start_evidence: {
        server_behind_baseline: pos.server_behind_baseline === true,
        server_on_correct_score_side: pos.server_on_correct_score_side === true,
        serving_partner_near_baseline: pos.serving_partner_near_baseline === true,
        receiver_behind_diagonal_baseline: pos.receiver_behind_diagonal_baseline === true,
        receiver_partner_near_kitchen: pos.receiver_partner_near_kitchen === true,
        diagonal_serve_detected: pos.diagonal_serve_detected === true,
      },
      end_evidence: {
        dead_ball_type: end.dead_ball_type || 'unknown',
        hand_pickup_detected: end.dead_ball_type === 'hand_pickup',
        ball_crossed_net_after_touch: false,
      },
      rally_winner_team: end.rally_winner_team || 'unknown',
      confidence,
      reviewStatus: confidence >= 0.75 && setupValid ? 'approved' : 'needs_review',
      included: endSeconds > startSeconds,
      reason: `${serve.notes || 'Serve detected.'} ${end.notes || 'Rally end analyzed.'}`.trim(),
    });

    state = advanceScoreState(state, end.rally_winner_team);
  }

  return {
    points,
    total_points_detected: points.length,
    approved_points: points.filter((p) => p.reviewStatus === 'approved').length,
    needs_review_points: points.filter((p) => p.reviewStatus !== 'approved').length,
    analysis_notes: `Autocut complete. Duration ${duration.toFixed(1)}s. Candidates ${candidates.length}. Valid serves ${validatedServes.length}. Points ${points.length}.`,
  };
}

function validateClips(clips) {
  if (!Array.isArray(clips) || clips.length === 0) {
    throw new Error('No clips supplied for export.');
  }

  return clips.map((clip, index) => {
    const start = Number(clip.startSeconds);
    const end = Number(clip.endSeconds);

    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) {
      throw new Error(`Invalid clip timing at clip ${index + 1}.`);
    }

    return {
      pointOrder: clip.pointOrder || index + 1,
      startSeconds: start,
      endSeconds: end,
      duration: end - start,
    };
  });
}

async function createExport(payload) {
  if (!payload.sourceVideoUrl) throw new Error('Missing sourceVideoUrl.');
  if (!CLOUDINARY_CLOUD_NAME || !CLOUDINARY_API_KEY || !CLOUDINARY_API_SECRET) {
    throw new Error('Missing Cloudinary env vars.');
  }

  const clips = validateClips(payload.clips);
  const jobId = randomUUID();
  const jobDir = path.join(EXPORT_DIR, jobId);
  await fs.mkdir(jobDir, { recursive: true });

  const sourcePath = path.join(jobDir, 'source.mp4');
  await downloadFile(payload.sourceVideoUrl, sourcePath);

  const clipPaths = [];

  for (let i = 0; i < clips.length; i++) {
    const clip = clips[i];
    const clipPath = path.join(jobDir, `clip-${String(i + 1).padStart(4, '0')}.mp4`);
    clipPaths.push(clipPath);

    await run('ffmpeg', [
      '-y',
      '-ss', String(clip.startSeconds),
      '-i', sourcePath,
      '-t', String(clip.duration),
      '-map', '0:v:0',
      '-map', '0:a?',
      '-c:v', 'libx264',
      '-preset', 'veryfast',
      '-crf', '20',
      '-c:a', 'aac',
      '-movflags', '+faststart',
      clipPath,
    ]);
  }

  const concatPath = path.join(jobDir, 'concat.txt');
  await fs.writeFile(
    concatPath,
    clipPaths.map((clipPath) => `file '${clipPath.replace(/'/g, "'\\''")}'`).join('\n')
  );

  const safeName = (payload.outputFilename || `picklecut-${jobId}.mp4`)
    .replace(/[^a-zA-Z0-9._-]/g, '-')
    .replace(/-+/g, '-');

  const outputPath = path.join(jobDir, safeName);

  await run('ffmpeg', [
    '-y',
    '-f', 'concat',
    '-safe', '0',
    '-i', concatPath,
    '-c', 'copy',
    '-movflags', '+faststart',
    outputPath,
  ]);

  const upload = await cloudinary.uploader.upload(outputPath, {
    resource_type: 'video',
    folder: 'picklecut/exports',
    public_id: `export-${jobId}`,
    overwrite: true,
  });

  return {
    outputUrl: upload.secure_url,
    downloadUrl: upload.secure_url,
    cloudinaryPublicId: upload.public_id,
    clipCount: clips.length,
    durationSeconds: clips.reduce((sum, clip) => sum + clip.duration, 0),
  };
}

function createCloudinaryUploadSignature() {
  if (!CLOUDINARY_API_SECRET || !CLOUDINARY_API_KEY || !CLOUDINARY_CLOUD_NAME) {
    throw new Error('Missing Cloudinary env vars.');
  }

  const timestamp = Math.floor(Date.now() / 1000);
  const folder = 'picklecut/source';

  const paramsToSign = `folder=${folder}&timestamp=${timestamp}${CLOUDINARY_API_SECRET}`;
  const signature = crypto.createHash('sha1').update(paramsToSign).digest('hex');

  return {
    cloudName: CLOUDINARY_CLOUD_NAME,
    apiKey: CLOUDINARY_API_KEY,
    timestamp,
    signature,
    folder,
  };
}

const server = http.createServer(async (req, res) => {
  if (req.method === 'OPTIONS') {
    sendJson(res, 204, {});
    return;
  }

  try {
    const url = new URL(req.url, PUBLIC_BASE_URL);

    if (req.method === 'GET' && url.pathname === '/health') {
      sendJson(res, 200, {
        ok: true,
        service: 'picklecut-backend',
        hasOpenAI: Boolean(OPENAI_API_KEY),
        hasCloudinary: Boolean(CLOUDINARY_CLOUD_NAME && CLOUDINARY_API_KEY && CLOUDINARY_API_SECRET),
      });
      return;
    }

    if (req.method === 'POST' && url.pathname === '/cloudinary/sign-upload') {
      sendJson(res, 200, createCloudinaryUploadSignature());
      return;
    }

    if (req.method === 'POST' && url.pathname === '/autocut') {
      const payload = await readJson(req);
      const result = await createAutocut(payload);
      sendJson(res, 200, result);
      return;
    }

    if (req.method === 'POST' && url.pathname === '/export') {
      const payload = await readJson(req);
      const result = await createExport(payload);
      sendJson(res, 200, result);
      return;
    }

    sendJson(res, 404, { error: 'Not found' });
  } catch (err) {
    sendJson(res, 500, {
      error: err.message || 'Server error',
    });
  }
});

await fs.mkdir(EXPORT_DIR, { recursive: true });
await fs.mkdir(WORK_DIR, { recursive: true });

server.listen(PORT, () => {
  console.log(`PickleCut backend running on port ${PORT}`);
});

