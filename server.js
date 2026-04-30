const autocutJobs = new Map();
const JOB_TIMEOUT_MS = 8 * 60 * 1000;

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
        progress: 5,
        message: 'Downloading Cloudinary video and preparing analysis.',
      });

      const result = await createAutocut(payload);

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

    // New async autocut API
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

    // Backward-compatible old sync endpoint
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
      path: url.pathname,
      method: req.method,
    });
  } catch (err) {
    sendJson(res, 500, {
      error: err.message || 'Server error',
    });
  }
});
