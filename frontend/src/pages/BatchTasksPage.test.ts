import { describe, expect, it } from "vitest";

import { parseWorkbook, pickField } from "./BatchTasksPage";

describe("BatchTasksPage helpers", () => {
  it("picks first available field from candidates", () => {
    const row = {
      Qsubj: "今年利润目标？",
      Reply: "请关注后续公告。",
    };
    expect(pickField(row, ["question", "Qsubj"])).toBe("今年利润目标？");
    expect(pickField(row, ["answer", "Reply"])).toBe("请关注后续公告。");
  });

  it("parses rows into batch infer items and filters invalid rows", () => {
    const rows = [
      { sample_id: "s1", question: "问题1", answer: "回答1" },
      { id: "s2", Qsubj: "问题2", Reply: "回答2", company_name: "测试公司" },
      { sample_id: "s3", question: "", answer: "回答3" },
    ];
    const parsed = parseWorkbook(rows);
    expect(parsed).toHaveLength(2);
    expect(parsed[0].sample_id).toBe("s1");
    expect(parsed[1].sample_id).toBe("s2");
    expect(parsed[1].company_name).toBe("测试公司");
  });
});
