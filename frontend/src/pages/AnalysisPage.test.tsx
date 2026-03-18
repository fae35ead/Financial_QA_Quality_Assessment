import { cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
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
  window.localStorage.clear();
});

describe("AnalysisPage", () => {
  it("renders inference result and allows manual enqueue", async () => {
    const mockFetch = vi.fn(async (input: string | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/infer")) {
        expect(init?.method).toBe("POST");
        return {
          ok: true,
          json: async () => ({
            sample_id: "sample-1",
            root_id: 2,
            root_label: "Evasive (mock)",
            root_confidence: 89.6,
            sub_label: "推迟回答",
            sub_confidence: 78.4,
            root_probabilities: {
              "Direct (mock)": 0.06,
              "Intermediate (mock)": 0.08,
              "Evasive (mock)": 0.86,
            },
            sub_probabilities: {
              推迟回答: 0.78,
              转移话题: 0.07,
            },
            entity_hits: [
              { text: "land", start: 2, end: 6, source_text: "question" },
              { text: "plan", start: 1, end: 5, source_text: "answer" },
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

    renderWithProviders(<AnalysisPage />);
    const [questionInput, answerInput] = screen.getAllByRole("textbox");
    fireEvent.change(questionInput, { target: { value: "question one" } });
    fireEvent.change(answerInput, { target: { value: "answer one" } });
    fireEvent.click(screen.getAllByRole("button")[0]);

    const evasiveNodes = await screen.findAllByText("逃避回答");
    expect(evasiveNodes.length).toBeGreaterThan(0);
    expect(screen.getByTestId("probability-radar")).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button")[1]);
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringMatching(/\/review\/sample-1\/enqueue$/),
        expect.objectContaining({ method: "POST" })
      );
    });
  });

  it("restores latest input and inference result after remount", async () => {
    const mockFetch = vi.fn(async (input: string | URL) => {
      const url = String(input);
      if (url.endsWith("/infer")) {
        return {
          ok: true,
          json: async () => ({
            sample_id: "sample-persist",
            root_id: 0,
            root_label: "Direct (mock)",
            root_confidence: 88.5,
            sub_label: "财务表现指引",
            sub_confidence: 77.2,
            root_probabilities: {
              "Direct (mock)": 0.85,
              "Intermediate (mock)": 0.1,
              "Evasive (mock)": 0.05,
            },
            sub_probabilities: {
              财务表现指引: 0.77,
              技术与研发进展: 0.23,
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

    const firstRender = renderWithProviders(<AnalysisPage />);
    const [firstQuestion, firstAnswer] = screen.getAllByRole("textbox");
    fireEvent.change(firstQuestion, { target: { value: "persisted question text" } });
    fireEvent.change(firstAnswer, { target: { value: "persisted answer text" } });
    fireEvent.click(screen.getAllByRole("button")[0]);

    const directNodes = await screen.findAllByText("直接响应");
    expect(directNodes.length).toBeGreaterThan(0);
    firstRender.unmount();

    renderWithProviders(<AnalysisPage />);
    const [restoredQuestion, restoredAnswer] = screen.getAllByRole("textbox");
    expect(restoredQuestion).toHaveValue("persisted question text");
    expect(restoredAnswer).toHaveValue("persisted answer text");
    expect(screen.getAllByText("直接响应").length).toBeGreaterThan(0);
  });
});

