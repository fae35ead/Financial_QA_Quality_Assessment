import type {
  AgentSuggestionResponse,
  AnnotatePayload,
  AnnotateResponse,
  BatchInferItem,
  BatchInferResponse,
  InferResponse,
  JobStatusResponse,
  ReviewDetailResponse,
  ReviewQueueResponse,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch (error) {
    const reason =
      error instanceof Error ? error.message : "network_error";
    throw new Error(
      `无法连接后端 ${API_BASE}（可能未启动或被 CORS 拦截）：${reason}`
    );
  }
  if (!response.ok) {
    let message = `请求失败: ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      // ignore JSON parse error and fallback to default message
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export const api = {
  infer(payload: { question: string; answer: string; company_name?: string; qa_time?: string | null }) {
    return request<InferResponse>("/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  enqueueReview(sampleId: string) {
    return request<{ sample_id: string; review_status: string; enqueued: boolean }>(`/review/${sampleId}/enqueue`, {
      method: "POST",
    });
  },

  batchInfer(items: BatchInferItem[]) {
    return request<BatchInferResponse>("/batch_infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });
  },

  getJob(jobId: string) {
    return request<JobStatusResponse>(`/jobs/${jobId}`);
  },

  async exportJob(jobId: string): Promise<Blob> {
    let response: Response;
    try {
      response = await fetch(`${API_BASE}/jobs/${jobId}/export`);
    } catch (error) {
      const reason =
        error instanceof Error ? error.message : "network_error";
      throw new Error(`无法连接后端 ${API_BASE}：${reason}`);
    }
    if (!response.ok) {
      let message = "导出失败";
      try {
        const payload = (await response.json()) as { detail?: string };
        if (payload.detail) {
          message = payload.detail;
        }
      } catch {
        // ignore
      }
      throw new Error(message);
    }
    return response.blob();
  },

  getReviewQueue(page = 1, pageSize = 20) {
    return request<ReviewQueueResponse>(`/review/queue?page=${page}&page_size=${pageSize}`);
  },

  getReviewDetail(sampleId: string) {
    return request<ReviewDetailResponse>(`/review/${sampleId}`);
  },

  requestAgentSuggestion(sampleId: string) {
    return request<AgentSuggestionResponse>(`/review/${sampleId}/agent-suggestion`, { method: "POST" });
  },

  annotate(payload: AnnotatePayload) {
    return request<AnnotateResponse>("/annotate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
};
