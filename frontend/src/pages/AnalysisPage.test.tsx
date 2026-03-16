import { cleanup, fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../components/ProbabilityRadar", () => ({
  ProbabilityRadar: ({ title }: { title: string }) => <div data-testid="probability-radar">{title}</div>,
}));

import { AnalysisPage } from "./AnalysisPage";
import { renderWithProviders } from "../test/renderWithProviders";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("AnalysisPage", () => {
  it("renders inference result with highlight and radar, then allows manual enqueue", async () => {
    const mockFetch = vi.fn(async (input: string | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/infer")) {
        expect(init?.method).toBe("POST");
        return {
          ok: true,
          json: async () => ({
            sample_id: "sample-1",
            root_id: 2,
            root_label: "Evasive (打太极)",
            root_confidence: 89.6,
            sub_label: "推迟回答",
            sub_confidence: 78.4,
            root_probabilities: {
              "Direct (直接响应)": 0.06,
              "Intermediate (避重就轻)": 0.08,
              "Evasive (打太极)": 0.86,
            },
            sub_probabilities: {
              推迟回答: 0.78,
              转移话题: 0.07,
              战略性模糊: 0.1,
              外部归因: 0.05,
            },
            entity_hits: [
              { text: "土地拍卖", start: 10, end: 14, source_text: "question" },
              { text: "拿地", start: 6, end: 8, source_text: "answer" },
            ],
            review_status: "pending_review",
            warning: null,
          }),
        };
      }
      if (url.endsWith("/review/sample-1/enqueue")) {
        expect(init?.method).toBe("POST");
        return {
          ok: true,
          json: async () => ({
            sample_id: "sample-1",
            review_status: "pending_review",
            enqueued: true,
          }),
        };
      }
      return {
        ok: false,
        json: async () => ({ detail: "not found" }),
      };
    });
    vi.stubGlobal("fetch", mockFetch);

    const { container } = renderWithProviders(<AnalysisPage />);
    fireEvent.change(screen.getByPlaceholderText("请输入提问文本"), {
      target: { value: "请问公司最近有没有参与土地拍卖计划？" },
    });
    fireEvent.change(screen.getByPlaceholderText("请输入回答文本"), {
      target: { value: "公司会在合适时机拿地，感谢关注。" },
    });
    fireEvent.click(screen.getByRole("button", { name: "开始分析" }));

    await screen.findByText("分析结果");
    expect(screen.getAllByText("Evasive (打太极)").length).toBeGreaterThan(0);
    expect(screen.getByTestId("probability-radar")).toBeInTheDocument();
    expect(container.querySelectorAll("mark.highlight-hit").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "手动加入待复核队列" }));
    await screen.findAllByText("复核状态");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/review\/sample-1\/enqueue$/),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("renders radar chart for direct responses when sub probabilities exist", async () => {
    const mockFetch = vi.fn(async (input: string | URL) => {
      const url = String(input);
      if (url.endsWith("/infer")) {
        return {
          ok: true,
          json: async () => ({
            sample_id: "sample-2",
            root_id: 0,
            root_label: "Direct (直接响应)",
            root_confidence: 91.2,
            sub_label: "财务表现指引",
            sub_confidence: 82.4,
            root_probabilities: {
              "Direct (直接响应)": 0.9,
              "Intermediate (避重就轻)": 0.06,
              "Evasive (打太极)": 0.04,
            },
            sub_probabilities: {
              财务表现指引: 0.82,
              技术与研发进展: 0.1,
              资本运作与并购: 0.08,
            },
            entity_hits: [],
            review_status: null,
            warning: null,
          }),
        };
      }
      return {
        ok: false,
        json: async () => ({ detail: "not found" }),
      };
    });
    vi.stubGlobal("fetch", mockFetch);

    renderWithProviders(<AnalysisPage />);
    fireEvent.change(screen.getByPlaceholderText("请输入提问文本"), {
      target: { value: "请问今年营收指引如何？" },
    });
    fireEvent.change(screen.getByPlaceholderText("请输入回答文本"), {
      target: { value: "公司将持续优化经营质量并关注收入增长。" },
    });
    fireEvent.click(screen.getByRole("button", { name: "开始分析" }));

    await screen.findByText("分析结果");
    expect(screen.getByText("直接回答子节点雷达图")).toBeInTheDocument();
    expect(screen.getByTestId("probability-radar")).toBeInTheDocument();
  });
});
