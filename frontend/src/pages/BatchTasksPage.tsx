import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Alert,
  Button,
  Card,
  Progress,
  Space,
  Table,
  Tag,
  Typography,
  Upload,
  message,
  type UploadProps,
} from "antd";
import { DownloadOutlined, UploadOutlined } from "@ant-design/icons";
import { useMemo, useState } from "react";
import * as XLSX from "xlsx";

import { api } from "../api/client";
import type { BatchInferItem, JobResultItem } from "../types";

const QUESTION_FIELDS = ["question", "Question", "Qsubj", "问题", "提问"];
const ANSWER_FIELDS = ["answer", "Answer", "Reply", "回答", "回复"];
const MAX_BATCH_ITEMS = 500;

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
  if (status === "completed") return <Tag color="success">completed</Tag>;
  if (status === "failed") return <Tag color="error">failed</Tag>;
  if (status === "running") return <Tag color="processing">running</Tag>;
  return <Tag>pending</Tag>;
}

export function BatchTasksPage() {
  const [items, setItems] = useState<BatchInferItem[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [messageApi, contextHolder] = message.useMessage();

  const jobQuery = useQuery({
    queryKey: ["job-status", jobId],
    queryFn: () => api.getJob(jobId ?? ""),
    enabled: Boolean(jobId),
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
      messageApi.success(`批任务已创建：${payload.job_id}`);
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
          throw new Error("文件中未找到工作表。");
        }
        const worksheet = workbook.Sheets[firstSheet];
        const rows = XLSX.utils.sheet_to_json<Record<string, unknown>>(worksheet, { defval: "" });
        const parsed = parseWorkbook(rows);
        if (!parsed.length) {
          throw new Error("未识别到有效问答列，请确认存在 question/answer 或 Qsubj/Reply。");
        }
        if (parsed.length > MAX_BATCH_ITEMS) {
          throw new Error(`单次批量上限为 ${MAX_BATCH_ITEMS} 条，请拆分后重试。`);
        }
        setItems(parsed);
        messageApi.success(`已解析 ${parsed.length} 条样本`);
      } catch (error) {
        const text = error instanceof Error ? error.message : "文件解析失败";
        messageApi.error(text);
      }
      return false;
    },
  };

  const tableData = useMemo<JobResultItem[]>(() => jobQuery.data?.results ?? [], [jobQuery.data?.results]);
  const exportDisabled = !jobId || jobQuery.data?.status !== "completed" || exportMutation.isPending;

  return (
    <div className="page-wrap fade-in">
      {contextHolder}
      <Card className="soft-card" title="批量任务页">
        <Space direction="vertical" style={{ width: "100%" }} size={16}>
          <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
            支持 CSV / Excel 文件，前端解析后调用 `/batch_infer` 提交任务。单次最多 500 条样本。
          </Typography.Paragraph>
          <Space wrap>
            <Upload {...uploadProps}>
              <Button icon={<UploadOutlined />}>上传 CSV/Excel</Button>
            </Upload>
            <Button type="primary" onClick={() => submitMutation.mutate()} disabled={!items.length || submitMutation.isPending}>
              创建批量任务
            </Button>
            <Button
              icon={<DownloadOutlined />}
              disabled={exportDisabled}
              loading={exportMutation.isPending}
              onClick={() => exportMutation.mutate()}
            >
              导出 CSV
            </Button>
          </Space>
          <Alert
            type="info"
            showIcon
            message={`当前已加载样本：${items.length} 条`}
            description={jobId ? `当前任务ID：${jobId}` : "尚未创建任务"}
          />
        </Space>
      </Card>

      {jobId && (
        <Card className="soft-card" title="任务进度">
          {jobQuery.data ? (
            <Space direction="vertical" style={{ width: "100%" }}>
              <Space>
                <Typography.Text strong>状态：</Typography.Text>
                {statusTag(jobQuery.data.status)}
              </Space>
              <Progress percent={jobQuery.data.progress} status={jobQuery.data.status === "failed" ? "exception" : "active"} />
              <Typography.Text type="secondary">
                完成 {jobQuery.data.completed}/{jobQuery.data.total}，失败 {jobQuery.data.failed}
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
            { title: "索引", dataIndex: "index", width: 90 },
            {
              title: "状态",
              dataIndex: "status",
              width: 120,
              render: (status: string) => statusTag(status),
            },
            {
              title: "根标签",
              render: (_, record) => record.result?.root_label ?? "-",
            },
            {
              title: "子标签",
              render: (_, record) => record.result?.sub_label ?? "-",
            },
            {
              title: "置信度",
              render: (_, record) =>
                record.result ? `${record.result.root_confidence}% / ${record.result.sub_confidence}%` : "-",
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
