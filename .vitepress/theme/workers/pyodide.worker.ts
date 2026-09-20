const PYODIDE_VERSION = '0.27.7'
const PYODIDE_BASE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

type RunMessage = {
  type: 'run'
  code: string
}

let pyodidePromise: Promise<any> | null = null

async function getPyodide() {
  if (!pyodidePromise) {
    self.postMessage({ type: 'status', status: 'loading' })
    pyodidePromise = import(/* @vite-ignore */ `${PYODIDE_BASE_URL}pyodide.mjs`)
      .then(({ loadPyodide }) => loadPyodide({ indexURL: PYODIDE_BASE_URL }))
    await pyodidePromise
    self.postMessage({ type: 'status', status: 'ready' })
  }
  return pyodidePromise
}

self.onmessage = async (event: MessageEvent<RunMessage>) => {
  if (event.data.type !== 'run') return

  try {
    const pyodide = await getPyodide()
    let stdout = ''
    let stderr = ''

    pyodide.setStdout({ batched: (text: string) => { stdout += `${text}\n` } })
    pyodide.setStderr({ batched: (text: string) => { stderr += `${text}\n` } })

    if (pyodide.globals.has('simulation')) {
      pyodide.globals.delete('simulation')
    }

    self.postMessage({ type: 'status', status: 'running' })
    await pyodide.runPythonAsync(event.data.code)

    let simulation: unknown = null
    if (pyodide.globals.has('simulation')) {
      const proxy = pyodide.globals.get('simulation')
      simulation = proxy?.toJs
        ? proxy.toJs({ dict_converter: Object.fromEntries })
        : proxy
      proxy?.destroy?.()
    }

    self.postMessage({
      type: 'result',
      stdout: stdout.trimEnd(),
      stderr: stderr.trimEnd(),
      simulation,
    })
  } catch (error) {
    self.postMessage({
      type: 'error',
      message: error instanceof Error ? error.message : String(error),
    })
  }
}

export {}
