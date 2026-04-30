import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { createWriteStream, promises as fs } from 'node:fs';
import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
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
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';

const autocutJobs = new Map();
const JOB_TIMEOUT_MS = 8 * 60 * 1000;

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
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`${command} exited ${code}: ${stderr.slice(-2500)}`));
      }
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
  const cleaned = String(text || '')
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

    throw new Error(`AI did not return valid JSON: ${cleaned.slice(0, 500)}`);
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
        detail: 'auto',
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

async function detectServeCandidates(videoPath, duration, jobDir, onProgress) {
  onProgress?.(20, 'Extracting coarse frames.');

  const framesDir = path.join(jobDir, 'coarse');

  // Scan dày hơn: mỗi 2 giây, tối đa 180 frame (~6 phút video).
  // Bản cũ mỗi 5 giây dễ bỏ lỡ cú giao thật và nhầm sang cú trả giao.
  const timestamps = makeRange(0, duration, 3).slice(0, 90);
  const frames = await extractFrames(videoPath, timestamps, framesDir, 'coarse', 480);

  const candidates = [];

  for (let i = 0; i < frames.length; i += 12) {
    onProgress?.(
      28 + Math.round((i / Math.max(frames.length, 1)) * 15),
      'Detecting true serve candidate windows.'
    );

    const batch = frames.slice(i, i + 12);

    const result = await callVisionJson({
      maxImages: 12,
      frames: batch,
      prompt: `
You are detecting TRUE pickleball doubles SERVE moments from sampled video frames.

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

A point starts ONLY at a legal serve contact, not at a return shot.

A TRUE serve candidate must show this sequence:
1. One player is behind the baseline and has/controls the ball before contact.
2. That same player performs a serve motion.
3. The ball leaves the server side after paddle contact.
4. The ball travels diagonally toward the opposite service box.
5. The receiver does NOT touch the ball until after the serve crosses the net.

Reject the timestamp if:
- the ball is already flying from the opposite court toward the hitter before contact
- the hitter is returning an incoming serve
- the contact is a return, drive, volley, dink, reset, or any mid-rally shot
- the player did not visibly possess/control the ball before contact
- players are walking into position, picking up the ball, passing the ball, or warming up

Be generous only for TRUE serve candidates. If uncertain whether the hitter possessed the ball before contact, use low confidence below 0.3.
`,
    });

    for (const item of result.serve_candidates || []) {
      if (Number(item.confidence || 0) >= 0.3) {
        candidates.push(Number(item.serve_contact_seconds));
      }
    }
  }

  return dedupeTimes(candidates, 5);
}
async function validateServe(videoPath, candidateTime, matchContext, scoreState, jobDir, index) {
  const start = Math.max(0, candidateTime - 3);
  const end = candidateTime + 3;

  // Scan mịn hơn quanh candidate để phân biệt serve và return.
  const timestamps = makeRange(start, end, 0.6);
  const frames = await extractFrames(videoPath, timestamps, path.join(jobDir, `serve-${index}`), 'serve', 720);

  const servingScore = scoreState.servingTeam === 'A' ? scoreState.scoreA : scoreState.scoreB;
  const scoreCall = scoreCallBeforeServe(scoreState);
  const expectedSide = expectedSideForServingScore(servingScore);

  return callVisionJson({
    maxImages: 10,
    frames,
    prompt: `
You are validating the TRUE start of a pickleball doubles point.

Match:
Team A: ${matchContext.teamA.name}
Team B: ${matchContext.teamB.name}

Expected score before this serve: ${scoreCall}
Expected serving team: Team ${scoreState.servingTeam}
Expected server number: ${scoreState.serverNumber}
Expected server side: ${expectedSide}

Find the exact TRUE serve contact timestamp.

CRITICAL RULE:
A legal serve contact is valid ONLY if the hitter possessed/controlled the ball immediately before contact.
If the ball is already coming from the opposite side toward the hitter, this is a RETURN, not a serve. Return serve_found=false.

A valid point-start serve must satisfy:
1. server_behind_baseline = true
2. server_possessed_ball_before_contact = true
3. server performs a serve motion from their own side
4. ball leaves the server side after contact
5. ball travels diagonally to the opposite service box
6. receiver is behind the diagonal opposite baseline
7. receiver partner is near the kitchen/NVZ line
8. serving partner is near or behind baseline, not at kitchen

Reject and return serve_found=false if:
- the contact is made by the receiving player after the ball crossed the net
- the hitter is returning an incoming serve
- the contact is any mid-rally shot
- the player did not visibly have/control the ball before contact
- the ball trajectory before contact comes from the opponent court
- players are only walking, picking up, rolling, bouncing, or passing the ball
- it is a practice swing or warm-up hit

Return JSON only:
{
  "serve_found": boolean,
  "serve_contact_seconds": number,
  "server_player": "A1" | "A2" | "B1" | "B2" | "unknown",
  "confidence": number,
  "player_positions": {
  "server_behind_baseline": boolean,
  "server_on_correct_score_side": boolean,
  "server_possessed_ball_before_contact": boolean,
  "incoming_ball_from_opponent_before_contact": boolean,
  "serving_partner_near_baseline": boolean,
  "receiver_behind_diagonal_baseline": boolean,
  "receiver_partner_near_kitchen": boolean,
  "diagonal_serve_detected": boolean
}
  },
  "notes": string
}

Important decision rule:
- Reject only if you can clearly see the ball came from the opponent court before this contact.
- If the ball before contact is too small or unclear, do NOT reject only because possession is unclear.
- If the player is behind baseline, in serve formation, and the ball travels diagonally after contact, return serve_found=true with lower confidence.
- Use confidence 0.45–0.70 when serve is likely but ball possession before contact is not clearly visible.

`,
  });
}

async function detectRallyEnd(videoPath, serveTime, nextServeTime, duration, jobDir, index) {
  const searchEnd = Math.min(duration, nextServeTime ? nextServeTime - 0.5 : serveTime + 30);
  const start = serveTime + 1;
  const timestamps = makeRange(start, searchEnd, 2).slice(0, 10);
  const frames = await extractFrames(videoPath, timestamps, path.join(jobDir, `end-${index}`), 'end', 720);

  const result = await callVisionJson({
    maxImages: 10,
    frames,
    prompt: `
You are detecting the exact END of a pickleball doubles point.

A serve happened at ${serveTime}s.

The clip must end at:
dead_ball_seconds + 0.2 second

Return JSON only:
{
  "dead_ball_found": boolean,
  "dead_ball_seconds": number,
  "dead_ball_type": "out_of_bounds" | "net_not_over" | "double_bounce" | "fault" | "hand_pickup" | "unknown",
  "rally_winner_team": "A" | "B" | "unknown",
  "confidence": number,
  "notes": string
}

A point ends at the EARLIEST clear dead-ball moment:
1. out_of_bounds
   - ball lands clearly outside the court lines
2. net_not_over
   - ball hits the net and does not cross to the opposite side
3. double_bounce
   - ball bounces twice before being returned
4. fault
   - clear visible fault and rally stops
5. hand_pickup
   - a player touches or picks up the dead ball by hand after the rally is over

Use hand_pickup only if earlier dead-ball evidence is unclear.

Do NOT end the clip only because:
- players slow down
- audio becomes quiet
- players reposition while the ball may still be live
- ball touches the net but crosses and rally continues
- camera view is temporarily unclear

If no clear dead-ball is visible, return dead_ball_found false.
`
,
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

async function createAutocut(payload, onProgress) {
  if (!payload.sourceVideoUrl) throw new Error('Missing sourceVideoUrl.');

  const internalJobId = randomUUID();
  const jobDir = path.join(WORK_DIR, internalJobId);
  await fs.mkdir(jobDir, { recursive: true });

  onProgress?.(8, 'Downloading Cloudinary video.');

  const sourcePath = path.join(jobDir, 'source.mp4');
  await downloadFile(payload.sourceVideoUrl, sourcePath);

  onProgress?.(14, 'Reading video duration.');

  const duration = await getVideoDuration(sourcePath);

  const matchContext = {
    teamA: payload.teamA || { name: 'Team A', players: {} },
    teamB: payload.teamB || { name: 'Team B', players: {} },
    firstServingTeam: payload.firstServingTeam || 'A',
  };

  const candidates = await detectServeCandidates(sourcePath, duration, jobDir, onProgress);

  onProgress?.(45, `Validating ${candidates.length} serve candidates.`);
let state = {
  scoreA: 0,
  scoreB: 0,
  servingTeam: payload.firstServingTeam || 'A',
  serverNumber: 2,
  serverPlayer: payload.firstServingTeam === 'B' ? 'B1' : 'A1',
};

  const validatedServes = [];

  for (let i = 0; i < candidates.length; i++) {
    onProgress?.(45 + Math.round((i / Math.max(candidates.length, 1)) * 20), `Validating serve ${i + 1}/${candidates.length}.`);

    const serve = await validateServe(sourcePath, candidates[i], matchContext, state, jobDir, i + 1).catch(() => null);

    if (serve?.serve_found && Number(serve.serve_contact_seconds) > 0 && Number(serve.confidence || 0) >= 0.2) {
      validatedServes.push(serve);
    }
  }

  validatedServes.sort((a, b) => Number(a.serve_contact_seconds) - Number(b.serve_contact_seconds));

  const points = [];

  for (let i = 0; i < validatedServes.length; i++) {
    onProgress?.(68 + Math.round((i / Math.max(validatedServes.length, 1)) * 22), `Detecting rally end ${i + 1}/${validatedServes.length}.`);

    const serve = validatedServes[i];
    const serveTime = Number(serve.serve_contact_seconds);
    const nextServe = validatedServes[i + 1];
    const nextServeTime = nextServe ? Number(nextServe.serve_contact_seconds) : null;

    const end = await detectRallyEnd(sourcePath, serveTime, nextServeTime, duration, jobDir, i + 1);

    const pos = serve.player_positions || {};
    const startSeconds = Math.max(0, serveTime - 1);
    const endSeconds = Math.min(duration, Number(end.dead_ball_seconds) + 0.2);
    const confidence = Math.min(Number(serve.confidence || 0.5), Number(end.confidence || 0.4));

  const isClearlyReturnShot = pos.incoming_ball_from_opponent_before_contact === true;

const setupValid =
  isClearlyReturnShot === false &&
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

  onProgress?.(95, 'Building point list.');

  return {
    points,
    total_points_detected: points.length,
    approved_points: points.filter((p) => p.reviewStatus === 'approved').length,
    needs_review_points: points.filter((p) => p.reviewStatus !== 'approved').length,
    analysis_notes: `Autocut complete. Duration ${duration.toFixed(1)}s. Candidates ${candidates.length}. Valid serves ${validatedServes.length}. Points ${points.length}.`,
  };
}

function getAutocutJob(jobId) {
  return autocutJobs.get(jobId);
}

function updateAutocutJob(jobId, patch) {
  const current = autocutJobs.get(jobId);
  if (!current) return;

  autocutJobs.set(jobId, {
    ...current,
    ...patch,
    updatedAt: new Date().toISOString(),
  });
}

function startAutocutJob(payload) {
  if (!payload.sourceVideoUrl) {
    throw new Error('Missing sourceVideoUrl.');
  }

  const jobId = randomUUID();

  autocutJobs.set(jobId, {
    id: jobId,
    status: 'queued',
    progress: 0,
    message: 'Queued autocut job.',
    result: null,
    error: null,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });

  setTimeout(() => {
    const job = autocutJobs.get(jobId);

    if (job && ['queued', 'processing'].includes(job.status)) {
      updateAutocutJob(jobId, {
        status: 'failed',
        progress: 100,
        message: 'Autocut job timed out.',
        error: 'Autocut job timed out after 8 minutes.',
      });
    }
  }, JOB_TIMEOUT_MS);

  queueMicrotask(async () => {
    try {
      updateAutocutJob(jobId, {
        status: 'processing',
        progress: 3,
        message: 'Starting autocut analysis.',
      });

      const result = await createAutocut(payload, (progress, message) => {
        updateAutocutJob(jobId, {
          status: 'processing',
          progress,
          message,
        });
      });

      updateAutocutJob(jobId, {
        status: 'complete',
        progress: 100,
        message: `Autocut complete. ${result.points?.length || 0} points detected.`,
        result,
        error: null,
      });
    } catch (err) {
      updateAutocutJob(jobId, {
        status: 'failed',
        progress: 100,
        message: 'Autocut failed.',
        error: err.message || 'Autocut failed.',
      });
    }
  });

  return getAutocutJob(jobId);
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
  const signature = createHash('sha1').update(paramsToSign).digest('hex');

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
        routes: [
          'GET /health',
          'POST /cloudinary/sign-upload',
          'POST /autocut/start',
          'GET /autocut/status/:jobId',
          'GET /autocut/result/:jobId',
          'POST /autocut',
          'POST /export',
        ],
      });
      return;
    }

    if (req.method === 'POST' && url.pathname === '/cloudinary/sign-upload') {
      sendJson(res, 200, createCloudinaryUploadSignature());
      return;
    }

    if (req.method === 'POST' && url.pathname === '/autocut/start') {
      const payload = await readJson(req);
      const job = startAutocutJob(payload);

      sendJson(res, 202, {
        jobId: job.id,
        status: job.status,
        progress: job.progress,
        message: job.message,
        statusUrl: `${PUBLIC_BASE_URL}/autocut/status/${job.id}`,
        resultUrl: `${PUBLIC_BASE_URL}/autocut/result/${job.id}`,
      });
      return;
    }

    if (req.method === 'GET' && url.pathname.startsWith('/autocut/status/')) {
      const jobId = url.pathname.split('/').pop();
      const job = getAutocutJob(jobId);

      if (!job) {
        sendJson(res, 404, { error: 'Autocut job not found.' });
        return;
      }

      sendJson(res, 200, {
        jobId: job.id,
        status: job.status,
        progress: job.progress,
        message: job.message,
        error: job.error,
        createdAt: job.createdAt,
        updatedAt: job.updatedAt,
      });
      return;
    }

    if (req.method === 'GET' && url.pathname.startsWith('/autocut/result/')) {
      const jobId = url.pathname.split('/').pop();
      const job = getAutocutJob(jobId);

      if (!job) {
        sendJson(res, 404, { error: 'Autocut job not found.' });
        return;
      }

      if (job.status === 'failed') {
        sendJson(res, 500, {
          jobId: job.id,
          status: job.status,
          error: job.error || 'Autocut failed.',
        });
        return;
      }

      if (job.status !== 'complete') {
        sendJson(res, 202, {
          jobId: job.id,
          status: job.status,
          progress: job.progress,
          message: job.message,
        });
        return;
      }

      sendJson(res, 200, job.result);
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

    sendJson(res, 404, {
      error: 'Not found',
      method: req.method,
      path: url.pathname,
    });
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
