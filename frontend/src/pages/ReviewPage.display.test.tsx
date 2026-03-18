import { cleanup, fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../test/renderWithProviders";
import { ReviewPage } from "./ReviewPage";

describe("ReviewPage display", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("hides pending_review tag in queue and shows model confidences in detail", async () => {
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
                question_text: "queue question",
                answer_text: "queue answer",
                layer1_label: "Evasive (mock)",
                layer1_confidence: 66.2,
                layer2_json: { sub_label: "推迟回答", sub_confidence: 55.4 },
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
            question_text: "queue question",
            answer_text: "queue answer",
            model_output: {
              layer1_label: "Evasive (mock)",
              layer1_confidence: 66.2,
              layer2_json: { sub_label: "推迟回答", sub_confidence: 55.4 },
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

    await screen.findByText("queue question");
    expect(screen.queryByText("pending_review")).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("queue question"));
    await screen.findByText("queue answer");
    expect(screen.getByText("根节点置信度：66.2%")).toBeInTheDocument();
    expect(screen.getByText("子节点置信度：55.4%")).toBeInTheDocument();
  });
});
