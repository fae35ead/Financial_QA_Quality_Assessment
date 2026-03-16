import { useMutation } from "@tanstack/react-query";
import { AuditOutlined } from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Col,
  Input,
  Progress,
  Row,
  Space,
  Statistic,
  Tag,
  Typography,
  message,
} from "antd";
import { useMemo, useState } from "react";

import { api } from "../api/client";
import { EntityHighlight } from "../components/EntityHighlight";
import { ProbabilityRadar } from "../components/ProbabilityRadar";
import type { InferResponse } from "../types";

const { TextArea } = Input;

const rootTagColor: Record<number, string> = {
  [-1]: "default",
  0: "success",
  1: "gold",
  2: "error",
};

const reviewStatusMeta: Record<string, { label: string; color: string }> = {
  pending_review: { label: "待复核", color: "processing" },
  reviewed: { label: "已复核", color: "success" },
  not_queued: { label: "未入队", color: "default" },
};

function probabilityList(probabilities: Record<string, number>) {
  return Object.entries(probabilities).sort((left, right) => right[1] - left[1]);
}

function getReviewStatus(status?: string | null) {
  if (!status) {
    return reviewStatusMeta.not_queued;
  }
  return reviewStatusMeta[status] ?? { label: status, color: "default" };
}

export function AnalysisPage() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [result, setResult] = useState<InferResponse | null>(null);
  const [messageApi, contextHolder] = message.useMessage();

  const inferMutation = useMutation({
    mutationFn: () => api.infer({ question, answer }),
    onSuccess: (payload) => {
      setResult(payload);
      if (payload.review_status === "pending_review") {
        messageApi.success("该样本已自动进入待复核队列。");
      }
    },
    onError: (error: Error) => {
      messageApi.error(error.message);
    },
  });

  const enqueueMutation = useMutation({
    mutationFn: (sampleId: string) => api.enqueueReview(sampleId),
    onSuccess: (payload) => {
      messageApi.success(payload.enqueued ? "已手动加入待复核队列。" : "该样本已在待复核队列中。");
    },
    onError: (error: Error) => {
      messageApi.error(error.message);
    },
  });

  const canSubmit = question.trim().length > 0 && answer.trim().length > 0 && !inferMutation.isPending;
  const readyForAnalysis = question.trim().length > 0 && answer.trim().length > 0;
  const completionRate = readyForAnalysis ? 100 : question.trim().length > 0 || answer.trim().length > 0 ? 50 : 0;
  const reviewStatus = getReviewStatus(result?.review_status);
  const strongestRootProbability = useMemo(() => {
    if (!result) {
      return null;
    }
    const values = Object.values(result.root_probabilities);
    if (!values.length) {
      return null;
    }
    return Number((Math.max(...values) * 100).toFixed(2));
  }, [result]);

  const radarProbabilities = useMemo(() => {
    if (!result) {
      return null;
    }
    if (!probabilityList(result.sub_probabilities).length) {
      return null;
    }
    return result.sub_probabilities;
  }, [result]);

  const radarTitle = useMemo(() => {
    if (!result) {
      return "子节点概率雷达图";
    }
    if (result.root_label.startsWith("Direct")) {
      return "直接回答子节点雷达图";
    }
    if (result.root_label.startsWith("Intermediate")) {
      return "部分响应子节点雷达图";
    }
    if (result.root_label.startsWith("Evasive")) {
      return "逃避战术雷达图";
    }
    return "子节点概率雷达图";
  }, [result]);

  return (
    <div className="page-wrap analysis-page fade-in">
      {contextHolder}
      <Card className="soft-card analysis-hero-card" variant="borderless">
        <Row gutter={[24, 20]} align="middle">
          <Col xs={24} xl={16}>
            <Space orientation="vertical" size={12} style={{ width: "100%" }}>
              <Tag className="analysis-eyebrow" icon={<AuditOutlined />}>
                首页分析
              </Tag>
              <Typography.Title level={2} className="analysis-hero-title">
                监管问答智能评估
              </Typography.Title>
              <Typography.Paragraph className="analysis-hero-desc">
                针对单条问答快速完成分类、置信度评估与复核入队判断，保证首屏可读与一步点击直达核心操作。
              </Typography.Paragraph>
              <div className="analysis-hero-stats">
                <div className="analysis-stat-item">
                  <span className="analysis-stat-label">输入完整度</span>
                  <strong>{completionRate}%</strong>
                </div>
                <div className="analysis-stat-item">
                  <span className="analysis-stat-label">复核状态</span>
                  <Tag color={reviewStatus.color}>{reviewStatus.label}</Tag>
                </div>
                <div className="analysis-stat-item">
                  <span className="analysis-stat-label">最高根节点概率</span>
                  <strong>{strongestRootProbability !== null ? `${strongestRootProbability}%` : "--"}</strong>
                </div>
              </div>
            </Space>
          </Col>
          <Col xs={24} xl={8}>
            <Space orientation="vertical" size={10} style={{ width: "100%" }}>
              <Button
                type="primary"
                size="large"
                onClick={() => inferMutation.mutate()}
                loading={inferMutation.isPending}
                disabled={!canSubmit}
                block
              >
                开始分析
              </Button>
              <Button
                size="large"
                onClick={() => result?.sample_id && enqueueMutation.mutate(result.sample_id)}
                disabled={!result?.sample_id || enqueueMutation.isPending}
                block
              >
                手动加入待复核队列
              </Button>
              <Typography.Text type="secondary" className="analysis-action-tip">
                建议先完成双栏输入，再执行分析。分析成功后可一键加入待复核队列。
              </Typography.Text>
            </Space>
          </Col>
        </Row>
      </Card>

      <Card className="soft-card analysis-input-card" title="输入区（左右分栏）">
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={12}>
            <div className="analysis-field-head">
              <Typography.Text strong>投资者提问</Typography.Text>
              <Typography.Text type="secondary">{question.length} 字</Typography.Text>
            </div>
            <TextArea
              className="analysis-textarea"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={8}
              placeholder="请输入提问文本"
            />
          </Col>
          <Col xs={24} lg={12}>
            <div className="analysis-field-head">
              <Typography.Text strong>董秘回答</Typography.Text>
              <Typography.Text type="secondary">{answer.length} 字</Typography.Text>
            </div>
            <TextArea
              className="analysis-textarea"
              value={answer}
              onChange={(event) => setAnswer(event.target.value)}
              rows={8}
              placeholder="请输入回答文本"
            />
          </Col>
        </Row>
        {!readyForAnalysis && (
          <Alert
            style={{ marginTop: 16 }}
            type="info"
            showIcon
            title="请输入完整问答文本后再开始分析。"
            banner
          />
        )}
      </Card>

      {result && (
        <Card className="soft-card analysis-result-card" title="分析结果">
          {result.warning && <Alert type="warning" showIcon message={result.warning} style={{ marginBottom: 16 }} />}
          <Row gutter={[16, 16]}>
            <Col xs={24} md={8}>
              <Card size="small" className="analysis-metric-card">
                <Space orientation="vertical">
                  <Tag color={rootTagColor[result.root_id] ?? "blue"}>{result.root_label}</Tag>
                  <Statistic title="根节点置信度" value={result.root_confidence} suffix="%" />
                  <Typography.Text type="secondary">{result.sub_label}</Typography.Text>
                </Space>
              </Card>
            </Col>
            <Col xs={24} md={8}>
              <Card size="small" className="analysis-metric-card">
                <Statistic title="子节点置信度" value={result.sub_confidence} suffix="%" />
                <Typography.Text type="secondary">样本ID：{result.sample_id ?? "-"}</Typography.Text>
              </Card>
            </Col>
            <Col xs={24} md={8}>
              <Card size="small" className="analysis-metric-card">
                <Typography.Text strong>复核状态</Typography.Text>
                <div style={{ marginTop: 10 }}>
                  <Tag color={reviewStatus.color}>{reviewStatus.label}</Tag>
                </div>
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]} style={{ marginTop: 8 }}>
            <Col xs={24} lg={12}>
              <Card size="small" className="analysis-inner-card" title="根节点概率分布">
                <Space orientation="vertical" style={{ width: "100%" }}>
                  {probabilityList(result.root_probabilities).map(([label, value]) => (
                    <div key={label} className="analysis-probability-item">
                      <Typography.Text>{label}</Typography.Text>
                      <Progress percent={Number((value * 100).toFixed(2))} size="small" showInfo />
                    </div>
                  ))}
                </Space>
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card size="small" className="analysis-inner-card" title="子节点概率分布">
                {probabilityList(result.sub_probabilities).length ? (
                  <Space orientation="vertical" style={{ width: "100%" }}>
                    {probabilityList(result.sub_probabilities).map(([label, value]) => (
                      <div key={label} className="analysis-probability-item">
                        <Typography.Text>{label}</Typography.Text>
                        <Progress percent={Number((value * 100).toFixed(2))} size="small" showInfo />
                      </div>
                    ))}
                  </Space>
                ) : (
                  <Typography.Text type="secondary">当前分支无子分类概率分布。</Typography.Text>
                )}
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]} style={{ marginTop: 8 }}>
            <Col xs={24} lg={12}>
              <Card size="small" className="analysis-inner-card" title="问题实体高亮">
                <Typography.Paragraph>
                  <EntityHighlight text={question} hits={result.entity_hits} source="question" />
                </Typography.Paragraph>
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card size="small" className="analysis-inner-card" title="回答实体高亮">
                <Typography.Paragraph>
                  <EntityHighlight text={answer} hits={result.entity_hits} source="answer" />
                </Typography.Paragraph>
              </Card>
            </Col>
          </Row>

          {radarProbabilities && (
            <Card size="small" className="analysis-inner-card" title={radarTitle} style={{ marginTop: 16 }}>
              <ProbabilityRadar title={`${result.root_label} 子节点概率分布`} probabilities={radarProbabilities} />
            </Card>
          )}
        </Card>
      )}
    </div>
  );
}
