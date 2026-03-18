import { cleanup, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { STORAGE_KEYS } from "../utils/persist";
import { renderWithProviders } from "../test/renderWithProviders";
import { BatchTasksPage } from "./BatchTasksPage";

describe("BatchTasksPage persistence", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it("restores latest job id and continues polling job status", async () => {
    window.localStorage.setItem(
      STORAGE_KEYS.batchPage,
      JSON.stringify({
        items: [{ sample_id: "s1", question: "q1", answer: "a1" }],
        jobId: "persist-job-1",
        jobSnapshot: {
          job_id: "persist-job-1",
          status: "running",
          total: 1,
          completed: 0,
          failed: 0,
          progress: 15,
          created_at: "2026-03-18T10:00:00Z",
          finished_at: null,
          results: [],
        },
      })
    );

    const mockFetch = vi.fn(async (input: string | URL) => {
      const url = String(input);
      if (url.endsWith("/jobs/persist-job-1")) {
        return {
          ok: true,
          json: async () => ({
            job_id: "persist-job-1",
            status: "completed",
            total: 1,
            completed: 1,
            failed: 0,
            progress: 100,
            created_at: "2026-03-18T10:00:00Z",
            finished_at: "2026-03-18T10:00:03Z",
            results: [
              {
                index: 0,
                sample_id: "s1",
                status: "completed",
                result: {
                  root_id: 2,
                  root_label: "Evasive (mock)",
                  root_confidence: 92.1,
                  sub_label: "推迟回答",
                  sub_confidence: 80.2,
                  root_probabilities: {},
                  sub_probabilities: {},
                  entity_hits: [],
                  warning: null,
                  sample_id: "s1",
                  is_low_confidence: false,
                  review_status: null,
                },
                error: null,
              },
            ],
          }),
        };
      }
      return {
        ok: false,
        json: async () => ({ detail: "not found" }),
      };
    });
    vi.stubGlobal("fetch", mockFetch);

    renderWithProviders(<BatchTasksPage />);

    await screen.findByText("逃避回答");
    expect(mockFetch).toHaveBeenCalledWith(expect.stringMatching(/\/jobs\/persist-job-1$/), undefined);
  });
});
