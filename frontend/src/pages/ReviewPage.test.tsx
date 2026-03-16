import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ReviewPage } from "./ReviewPage";
import { renderWithProviders } from "../test/renderWithProviders";

describe("ReviewPage", () => {
  it("loads queue and renders selected sample detail", async () => {
    const mockFetch = vi.fn(async (input: string | URL) => {
      const url = String(input);
      if (url.includes("/review/queue")) {
        return {
          ok: true,
          json: async () => ({
            page: 1,
            page_size: 50,
            total: 1,
            items: [
              {
                sample_id: "sample-1",
                question_text: "公司利润是否承压？",
                answer_text: "请关注后续公告",
                layer1_label: "Evasive (打太极)",
                layer1_confidence: 66.2,
                layer2_json: { sub_label: "推迟回答" },
                review_status: "pending_review",
                is_low_confidence: true,
                processed_at: "2026-03-15T00:00:00",
              },
            ],
          }),
        };
      }
      if (url.includes("/review/sample-1")) {
        return {
          ok: true,
          json: async () => ({
            sample_id: "sample-1",
            question_text: "公司利润是否承压？",
            answer_text: "请关注后续公告",
            model_output: {
              layer1_label: "Evasive (打太极)",
              layer1_confidence: 66.2,
              layer2_json: { sub_label: "推迟回答" },
              is_low_confidence: true,
              review_status: "pending_review",
              processed_at: "2026-03-15T00:00:00",
            },
            agent_suggestion: null,
            human_annotation: null,
          }),
        };
      }
      return {
        ok: false,
        json: async () => ({ detail: "not found" }),
      };
    });
    vi.stubGlobal("fetch", mockFetch);

    renderWithProviders(<ReviewPage />);
    await screen.findByText("公司利润是否承压？");
    fireEvent.click(screen.getByText("公司利润是否承压？"));
    await screen.findByText("请关注后续公告");
    expect(screen.getByRole("button", { name: "请求 Agent 建议" })).toBeInTheDocument();
  });
});
