import ReactECharts from "echarts-for-react";

interface ProbabilityRadarProps {
  title: string;
  probabilities: Record<string, number>;
}

export function ProbabilityRadar({ title, probabilities }: ProbabilityRadarProps) {
  const labels = Object.keys(probabilities);
  const values = labels.map((label) => Number((probabilities[label] * 100).toFixed(2)));
  if (!labels.length) {
    return null;
  }

  const option = {
    tooltip: {
      trigger: "item",
      formatter: (params: { value: number[] }) =>
        params.value
          .map((value, index) => `${labels[index]}: ${value.toFixed(2)}%`)
          .join("<br/>"),
    },
    radar: {
      radius: "65%",
      indicator: labels.map((label) => ({ name: label, max: 100 })),
      splitLine: {
        lineStyle: { color: "rgba(148, 163, 184, 0.4)" },
      },
      splitArea: {
        areaStyle: {
          color: ["rgba(37, 99, 235, 0.04)", "rgba(2, 132, 199, 0.03)"],
        },
      },
    },
    series: [
      {
        type: "radar",
        name: title,
        data: [
          {
            value: values,
            areaStyle: { color: "rgba(37, 99, 235, 0.2)" },
            lineStyle: { color: "#2563eb", width: 2 },
            itemStyle: { color: "#1d4ed8" },
          },
        ],
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: 300 }} />;
}
