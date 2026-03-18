import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, Card, Col, Form, Input, InputNumber, List, Row, Select, Space, Typography, message } from "antd";
import { useEffect, useMemo, useState } from "react";

import { api } from "../api/client";

const ROOT_OPTIONS = ["Direct (直接响应)", "Intermediate (避重就轻)", "Evasive (打太极)"];

const SUB_OPTIONS: Record<string, string[]> = {
  "Direct (直接响应)": ["资本运作与并购", "技术与研发进展", "产能与项目规划", "合规与风险披露", "财务表现指引"],
  "Intermediate (避重就轻)": ["无下游细分(部分响应)"],
  "Evasive (打太极)": ["推迟回答", "转移话题", "战略性模糊", "外部归因"],
};

function displayRootLabel(label?: string | null) {
  if (!label) return "-";
  const normalized = label.toLowerCase();
  if (normalized.startsWith("direct")) return "直接响应";
  if (normalized.startsWith("intermediate")) return "部分响应";
  if (normalized.startsWith("evasive")) return "逃避回答";
  return label;
}

export function ReviewPage() {
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [agentJobId, setAgentJobId] = useState<string | null>(null);
  const [messageApi, contextHolder] = message.useMessage();
  const [form] = Form.useForm();
  const queryClient = useQueryClient();

  const queueQuery = useQuery({
    queryKey: ["review-queue"],
    queryFn: () => api.getReviewQueue(1, 50),
    refetchInterval: 6000,
  });

  const detailQuery = useQuery({
    queryKey: ["review-detail", selectedSampleId],
    queryFn: () => api.getReviewDetail(selectedSampleId ?? ""),
    enabled: Boolean(selectedSampleId),
  });

  const agentJobQuery = useQuery({
    queryKey: ["agent-job", agentJobId],
    queryFn: () => api.getJob(agentJobId ?? ""),
    enabled: Boolean(agentJobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (!status || status === "pending" || status === "running") {
        return 1200;
      }
      return false;
    },
  });

  useEffect(() => {
    const status = agentJobQuery.data?.status;
    if (!status) {
      return;
    }
    if (status === "completed") {
      messageApi.success("Agent 建议已更新。");
      queryClient.invalidateQueries({ queryKey: ["review-detail", selectedSampleId] });
      queryClient.invalidateQueries({ queryKey: ["review-queue"] });
      setAgentJobId(null);
    } else if (status === "failed") {
      messageApi.error("Agent 建议任务执行失败。");
      setAgentJobId(null);
    }
  }, [agentJobQuery.data?.status, messageApi, queryClient, selectedSampleId]);

  const askAgentMutation = useMutation({
    mutationFn: () => api.requestAgentSuggestion(selectedSampleId ?? ""),
    onSuccess: (payload) => {
      setAgentJobId(payload.job_id);
    },
    onError: (error: Error) => {
      messageApi.error(error.message);
    },
  });

  const annotateMutation = useMutation({
    mutationFn: (payload: { root_label: string; sub_label: string; note?: string; annotator_confidence?: number | null }) =>
      api.annotate({
        sample_id: selectedSampleId ?? "",
        root_label: payload.root_label,
        sub_label: payload.sub_label,
        note: payload.note,
        annotator_id: "human_reviewer",
        annotator_confidence: payload.annotator_confidence,
      }),
    onSuccess: (payload) => {
      messageApi.success(`提交成功：${payload.review_status}`);
      queryClient.invalidateQueries({ queryKey: ["review-queue"] });
      queryClient.invalidateQueries({ queryKey: ["review-detail", selectedSampleId] });
    },
    onError: (error: Error) => {
      messageApi.error(error.message);
    },
  });

  useEffect(() => {
    if (!detailQuery.data) {
      return;
    }
    const modelOutput = detailQuery.data.model_output;
    const modelSub = String(modelOutput.layer2_json.sub_label ?? "");
    form.setFieldsValue({
      root_label: detailQuery.data.human_annotation?.root_label ?? modelOutput.layer1_label,
      sub_label: detailQuery.data.human_annotation?.sub_label ?? modelSub,
      note: detailQuery.data.human_annotation?.note ?? "",
      annotator_confidence: undefined,
    });
  }, [detailQuery.data, form]);

  const rootValue = Form.useWatch("root_label", form) as string | undefined;
  const subOptions = useMemo(() => SUB_OPTIONS[rootValue ?? ROOT_OPTIONS[0]] ?? [], [rootValue]);

  return (
    <div className="page-wrap fade-in">
      {contextHolder}
      <Row gutter={[16, 16]} className="review-layout">
        <Col xs={24} lg={8} className="review-col">
          <Card className="soft-card review-panel-card review-queue-card" title="待复核队列">
            <List
              className="review-queue-list"
              loading={queueQuery.isLoading}
              dataSource={queueQuery.data?.items ?? []}
              renderItem={(item) => (
                <List.Item
                  onClick={() => setSelectedSampleId(item.sample_id)}
                  style={{
                    cursor: "pointer",
                    borderRadius: 8,
                    padding: "10px 8px",
                    background: selectedSampleId === item.sample_id ? "rgba(37, 99, 235, 0.08)" : undefined,
                  }}
                >
                  <Space direction="vertical" style={{ width: "100%" }}>
                    <Typography.Text type="secondary">{displayRootLabel(item.layer1_label)}</Typography.Text>
                    <Typography.Text ellipsis>{item.question_text}</Typography.Text>
                  </Space>
                </List.Item>
              )}
            />
          </Card>
        </Col>

        <Col xs={24} lg={16} className="review-col">
          <Card
            className="soft-card review-panel-card review-detail-card"
            title="复核详情"
            extra={
              <Space>
                <Button onClick={() => askAgentMutation.mutate()} disabled={!selectedSampleId || askAgentMutation.isPending || Boolean(agentJobId)}>
                  请求 Agent 建议
                </Button>
              </Space>
            }
          >
            {detailQuery.data ? (
              <div className="review-detail-scroll">
                <Space direction="vertical" style={{ width: "100%" }} size={14}>
                  <Card size="small" title="样本文本">
                    <Typography.Paragraph>
                      <Typography.Text strong>问题：</Typography.Text> {detailQuery.data.question_text}
                    </Typography.Paragraph>
                    <Typography.Paragraph style={{ marginBottom: 0 }}>
                      <Typography.Text strong>回答：</Typography.Text> {detailQuery.data.answer_text}
                    </Typography.Paragraph>
                  </Card>
                  <Row gutter={[12, 12]}>
                    <Col xs={24} md={8}>
                      <Card size="small" title="模型结论">
                        <Typography.Text>{displayRootLabel(detailQuery.data.model_output.layer1_label)}</Typography.Text>
                        <br />
                        <Typography.Text type="secondary">{String(detailQuery.data.model_output.layer2_json.sub_label ?? "-")}</Typography.Text>
                        <br />
                        <Typography.Text type="secondary">根节点置信度：{detailQuery.data.model_output.layer1_confidence}%</Typography.Text>
                        <br />
                        <Typography.Text type="secondary">
                          子节点置信度：
                          {typeof detailQuery.data.model_output.layer2_json.sub_confidence === "number"
                            ? `${detailQuery.data.model_output.layer2_json.sub_confidence}%`
                            : "-"}
                        </Typography.Text>
                      </Card>
                    </Col>
                    <Col xs={24} md={8}>
                      <Card size="small" title="Agent 建议">
                        <Typography.Text>{displayRootLabel(detailQuery.data.agent_suggestion?.root_label)}</Typography.Text>
                        <br />
                        <Typography.Text type="secondary">{detailQuery.data.agent_suggestion?.sub_label ?? "-"}</Typography.Text>
                        <br />
                        <Typography.Text type="secondary">{detailQuery.data.agent_suggestion?.reason ?? "-"}</Typography.Text>
                      </Card>
                    </Col>
                    <Col xs={24} md={8}>
                      <Card size="small" title="人工复核">
                        <Form
                          form={form}
                          layout="vertical"
                          onFinish={(values: { root_label: string; sub_label: string; note?: string; annotator_confidence?: number }) =>
                            annotateMutation.mutate({
                              root_label: values.root_label,
                              sub_label: values.sub_label,
                              note: values.note,
                              annotator_confidence: typeof values.annotator_confidence === "number" ? values.annotator_confidence : null,
                            })
                          }
                        >
                          <Form.Item label="根标签" name="root_label" rules={[{ required: true }]}>
                            <Select options={ROOT_OPTIONS.map((item) => ({ label: displayRootLabel(item), value: item }))} />
                          </Form.Item>
                          <Form.Item label="子标签" name="sub_label" rules={[{ required: true }]}>
                            <Select options={subOptions.map((item) => ({ label: item, value: item }))} />
                          </Form.Item>
                          <Form.Item label="人工置信度 (0-1)" name="annotator_confidence">
                            <InputNumber style={{ width: "100%" }} min={0} max={1} step={0.01} />
                          </Form.Item>
                          <Form.Item label="备注" name="note">
                            <Input.TextArea rows={2} />
                          </Form.Item>
                          <Button type="primary" htmlType="submit" loading={annotateMutation.isPending}>
                            提交复核
                          </Button>
                        </Form>
                      </Card>
                    </Col>
                  </Row>
                </Space>
              </div>
            ) : (
              <Typography.Text type="secondary">请从左侧队列选择一个样本。</Typography.Text>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
}
