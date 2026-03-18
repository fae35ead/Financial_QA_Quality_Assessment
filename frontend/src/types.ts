export type SourceText = "question" | "answer";

export interface EntityHit {
  text: string;
  start: number;
  end: number;
  source_text: SourceText;
}

export interface InferResponse {
  root_id: number;
  root_label: string;
  root_confidence: number;
  sub_label: string;
  sub_confidence: number;
  root_probabilities: Record<string, number>;
  sub_probabilities: Record<string, number>;
  entity_hits: EntityHit[];
  warning?: string | null;
  sample_id?: string | null;
  is_low_confidence?: boolean | null;
  review_status?: string | null;
}

export interface BatchInferItem {
  sample_id?: string;
  company_name?: string;
  qa_time?: string | null;
  question: string;
  answer: string;
}

export interface BatchInferResponse {
  job_id: string;
  status: string;
  total: number;
}

export interface JobResultItem {
  index: number;
  sample_id?: string | null;
  status: string;
  result?: InferResponse | null;
  error?: string | null;
}

export interface JobStatusResponse {
  job_id: string;
  status: string;
  total: number;
  completed: number;
  failed: number;
  progress: number;
  created_at: string;
  finished_at?: string | null;
  results: JobResultItem[];
}

export interface ReviewQueueItem {
  sample_id: string;
  company_name?: string | null;
  qa_time?: string | null;
  question_text: string;
  answer_text: string;
  layer1_label: string;
  layer1_confidence: number;
  layer2_json: {
    sub_label?: string;
    sub_confidence?: number;
    warning?: string | null;
    [key: string]: unknown;
  };
  review_status?: string | null;
  is_low_confidence: boolean;
  processed_at: string;
}

export interface ReviewQueueResponse {
  page: number;
  page_size: number;
  total: number;
  items: ReviewQueueItem[];
}

export interface ReviewDetailResponse {
  sample_id: string;
  company_name?: string | null;
  qa_time?: string | null;
  question_text: string;
  answer_text: string;
  model_output: {
    layer1_label: string;
    layer1_confidence: number;
    layer2_json: {
      sub_label?: string;
      sub_confidence?: number;
      warning?: string | null;
      [key: string]: unknown;
    };
    is_low_confidence: boolean;
    review_status?: string | null;
    processed_at: string;
  };
  agent_suggestion?: {
    root_label?: string;
    sub_label?: string;
    confidence?: number;
    reason?: string;
  } | null;
  human_annotation?: {
    root_label?: string;
    sub_label?: string;
    note?: string;
  } | null;
}

export interface AgentSuggestionResponse {
  job_id: string;
  status: string;
  sample_id: string;
}

export interface AnnotatePayload {
  sample_id: string;
  root_label: string;
  sub_label: string;
  note?: string;
  annotator_id?: string;
  annotator_confidence?: number | null;
}

export interface AnnotateResponse {
  sample_id: string;
  review_status: string;
  annotation_id: string;
  training_corpus_file: string;
}
