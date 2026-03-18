import { useMutation, useQuery } from "@tanstack/react-query";
import { DownloadOutlined, UploadOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Progress, Space, Table, Tag, Typography, Upload, message, type UploadProps } from "antd";
import { useEffect, useMemo, useState } from "react";
import * as XLSX from "xlsx";

import { api } from "../api/client";
import type { BatchInferItem, JobResultItem, JobStatusResponse } from "../types";
import { STORAGE_KEYS, readLocalState, writeLocalState } from "../utils/persist";

const QUESTION_FIELDS = ["question", "Question", "Qsubj", "问题", "提问"];
const ANSWER_FIELDS = ["answer", "Answer", "Reply", "回答", "回复"];
const MAX_BATCH_ITEMS = 500;

type BatchPersistedState = {
  items: BatchInferItem[];
  jobId: string | null;
  jobSnapshot: JobStatusResponse | null;
};

function getBatchInitialState(): BatchPersistedState {
  const raw = readLocalState<Partial<BatchPersistedState>>(STORAGE_KEYS.batchPage, {});
  return {
    items: Array.isArray(raw.items) ? raw.items : [],
    jobId: typeof raw.jobId === "string" && raw.jobId.length > 0 ? raw.jobId : null,
    jobSnapshot: raw.jobSnapshot && typeof raw.jobSnapshot === "object" ? (raw.jobSnapshot as JobStatusResponse) : null,
  };
}

export function pickField(record: Record<string, unknown>, candidates: string[]): string {
  for (const candidate of candidates) {
    const raw = record[candidate];
    if (typeof raw === "string" && raw.trim()) {
      return raw.trim();
    }
  }
  return "";
}

export function parseWorkbook(rows: Record<string, unknown>[]): BatchInferItem[] {
  const parsed: BatchInferItem[] = [];
  rows.forEach((row, index) => {
    const question = pickField(row, QUESTION_FIELDS);
    const answer = pickField(row, ANSWER_FIELDS);
    if (!question || !answer) {
      return;
    }
    parsed.push({
      sample_id: String(row.sample_id ?? row.id ?? `row_${index + 1}`),
      company_name: typeof row.company_name === "string" ? row.company_name : undefined,
      question,
      answer,
    });
  });
  return parsed;
}

function statusTag(status: string) {
  if (status === "completed") return <Tag color="success">已完成</Tag>;
  if (status === "failed") return <Tag color="error">失败</Tag>;
  if (status === "running") return <Tag color="processing">运行中</Tag>;
  return <Tag>待执行</Tag>;
}

function displayRootLabel(label?: string | null) {
  if (!label) return "-";
  const normalized = label.toLowerCase();
  if (normalized.startsWith("direct")) return "直接响应";
  if (normalized.startsWith("intermediate")) return "部分响应";
  if (normalized.startsWith("evasive")) return "逃避回答";
  return label;
}

export function BatchTasksPage() {
  const initial = useMemo(() => getBatchInitialState(), []);
  const [items, setItems] = useState<BatchInferItem[]>(initial.items);
  const [jobId, setJobId] = useState<string | null>(initial.jobId);
  const [jobSnapshot, setJobSnapshot] = useState<JobStatusResponse | null>(initial.jobSnapshot);
  const [messageApi, contextHolder] = message.useMessage();

  const jobQuery = useQuery({
    queryKey: ["job-status", jobId],
    queryFn: () => api.getJob(jobId ?? ""),
    enabled: Boolean(jobId),
    initialData: jobSnapshot ?? undefined,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (!status || status === "pending" || status === "running") {
        return 1500;
      }
      return false;
    },
  });

  const submitMutation = useMutation({
    mutationFn: () => api.batchInfer(items),
    onSuccess: (payload) => {
      setJobId(payload.job_id);
      setJobSnapshot(null);
      messageApi.success(`批量任务已创建：${payload.job_id}`);
    },
    onError: (error: Error) => {
      messageApi.error(error.message);
    },
  });

  const exportMutation = useMutation({
    mutationFn: () => api.exportJob(jobId ?? ""),
    onSuccess: (blob) => {
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `batch_job_${jobId}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    },
    onError: (error: Error) => {
      messageApi.error(error.message);
    },
  });

  const uploadProps: UploadProps = {
    multiple: false,
    showUploadList: false,
    beforeUpload: async (file) => {
      try {
        const buffer = await file.arrayBuffer();
        const workbook = XLSX.read(buffer, { type: "array" });
        const firstSheet = workbook.SheetNames[0];
        if (!firstSheet) {
          throw new Error("上传文件中未找到工作表。");
        }
        const worksheet = workbook.Sheets[firstSheet];
        const rows = XLSX.utils.sheet_to_json<Record<string, unknown>>(worksheet, { defval: "" });
        const parsed = parseWorkbook(rows);
        if (!parsed.length) {
          throw new Error("未识别到有效的问题/回答列。");
        }
        if (parsed.length > MAX_BATCH_ITEMS) {
          throw new Error(`单次最多支持 ${MAX_BATCH_ITEMS} 条样本。`);
        }
        setItems(parsed);
        messageApi.success(`已加载 ${parsed.length} 条样本。`);
      } catch (error) {
        const text = error instanceof Error ? error.message : "文件解析失败。";
        messageApi.error(text);
      }
      return false;
    },
  };

  const activeJobData = jobQuery.data ?? jobSnapshot;
  const tableData = useMemo<JobResultItem[]>(() => activeJobData?.results ?? [], [activeJobData?.results]);
  const exportDisabled = !jobId || activeJobData?.status !== "completed" || exportMutation.isPending;

  useEffect(() => {
    if (jobQuery.data) {
      setJobSnapshot(jobQuery.data);
    }
  }, [jobQuery.data]);

  useEffect(() => {
    writeLocalState(STORAGE_KEYS.batchPage, { items, jobId, jobSnapshot });
  }, [items, jobId, jobSnapshot]);

  return (
    <div className="page-wrap fade-in">
      {contextHolder}
      <Card className="soft-card" title="批量任务">
        <Space direction="vertical" style={{ width: "100%" }} size={16}>
          <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
            支持上传 CSV/Excel 并创建批量推理任务，单次最多 500 条。
          </Typography.Paragraph>
          <Space wrap>
            <Upload {...uploadProps}>
              <Button icon={<UploadOutlined />}>上传 CSV/Excel</Button>
            </Upload>
            <Button type="primary" onClick={() => submitMutation.mutate()} disabled={!items.length || submitMutation.isPending}>
              创建批量任务
            </Button>
            <Button icon={<DownloadOutlined />} disabled={exportDisabled} loading={exportMutation.isPending} onClick={() => exportMutation.mutate()}>
              导出 CSV
            </Button>
          </Space>
          <Alert
            type="info"
            showIcon
            message={`已加载样本：${items.length}`}
            description={jobId ? `当前任务 ID：${jobId}` : "尚未创建批量任务。"}
          />
        </Space>
      </Card>

      {jobId && (
        <Card className="soft-card" title="任务进度">
          {activeJobData ? (
            <Space direction="vertical" style={{ width: "100%" }}>
              <Space>
                <Typography.Text strong>状态：</Typography.Text>
                {statusTag(activeJobData.status)}
              </Space>
              <Progress percent={activeJobData.progress} status={activeJobData.status === "failed" ? "exception" : "active"} />
              <Typography.Text type="secondary">
                已完成 {activeJobData.completed}/{activeJobData.total}，失败 {activeJobData.failed}
              </Typography.Text>
            </Space>
          ) : (
            <Typography.Text type="secondary">任务状态加载中...</Typography.Text>
          )}
        </Card>
      )}

      <Card className="soft-card" title="批量结果">
        <Table<JobResultItem>
          rowKey={(record) => `${record.index}-${record.sample_id ?? "none"}`}
          pagination={{ pageSize: 8 }}
          dataSource={tableData}
          columns={[
            { title: "序号", dataIndex: "index", width: 90 },
            {
              title: "状态",
              dataIndex: "status",
              width: 120,
              render: (status: string) => statusTag(status),
            },
            {
              title: "根标签",
              render: (_, record) => displayRootLabel(record.result?.root_label),
            },
            {
              title: "子标签",
              render: (_, record) => record.result?.sub_label ?? "-",
            },
            {
              title: "置信度",
              render: (_, record) => (record.result ? `${record.result.root_confidence}% / ${record.result.sub_confidence}%` : "-"),
            },
            {
              title: "错误信息",
              dataIndex: "error",
              render: (error?: string | null) => error ?? "-",
            },
          ]}
        />
      </Card>
    </div>
  );
}
