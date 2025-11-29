export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || 'HTTP error');
  }
  return response.json();
}

export async function uploadFile(file: File): Promise<{ filename: string; size: number; message: string }> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });
  
  return handleResponse<{ filename: string; size: number; message: string }>(response);
}

export async function ingestDocuments(options?: { fast_pass?: boolean }): Promise<{ status: string; documents_processed: number; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/ingest`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(options || {}),
  });
  
  return handleResponse<{ status: string; documents_processed: number; message: string }>(response);
}

export async function buildIndex(): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/build-index`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({}),
  });
  
  return handleResponse<{ status: string; message: string }>(response);
}

export async function query(params: {
  query: string;
  top_k?: number;
  use_reranker?: boolean;
}): Promise<{
  answer: string;
  sources: Array<{ content: string; score: number; metadata: Record<string, unknown> }>;
  trace_id: string;
  query_scrubbed: boolean;
}> {
  const { query: queryText, top_k = 5, use_reranker = true } = params;
  const response = await fetch(`${API_BASE_URL}/api/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query: queryText, top_k, use_reranker }),
  });
  
  return handleResponse<{
    answer: string;
    sources: Array<{ content: string; score: number; metadata: Record<string, unknown> }>;
    trace_id: string;
    query_scrubbed: boolean;
  }>(response);
}

export async function getDocuments(): Promise<Array<{ name: string; size: string; uploadDate: string }>> {
  const response = await fetch(`${API_BASE_URL}/api/documents`);
  return handleResponse<Array<{ name: string; size: string; uploadDate: string }>>(response);
}

export interface ReadinessResponse {
  components: {
    api: boolean;
    app: boolean;
    service_manager: boolean;
    retriever: boolean;
    generator: boolean;
    doc_loader: boolean;
    vector_store: boolean;
  };
  trace_id: string;
}

export async function initialize(): Promise<{ status: string; trace_id: string }> {
  const response = await fetch(`${API_BASE_URL}/api/initialize`, {
    method: 'POST',
  });
  return handleResponse<{ status: string; trace_id: string }>(response);
}

export async function getReadiness(): Promise<ReadinessResponse> {
  const response = await fetch(`${API_BASE_URL}/api/ready`);
  return handleResponse<ReadinessResponse>(response);
}

export async function getHealth(): Promise<{
  status: string;
  version: string;
  components: Record<string, string>;
}> {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  return handleResponse<{
    status: string;
    version: string;
    components: Record<string, string>;
  }>(response);
}

export async function warmOllama(): Promise<{ status: string; trace_id: string }> {
  const response = await fetch(`${API_BASE_URL}/api/warm-ollama`);
  return handleResponse<{ status: string; trace_id: string }>(response);
}

export async function getTrace(traceId: string): Promise<{ trace_id: string; events: Array<Record<string, unknown>> }> {
  const response = await fetch(`${API_BASE_URL}/api/traces/${traceId}`);
  return handleResponse<{ trace_id: string; events: Array<Record<string, unknown>> }>(response);
}
