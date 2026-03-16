import { ConfigProvider, theme } from "antd";
import zhCN from "antd/locale/zh_CN";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";

import App from "./App";
import "./index.css";

const queryClient = new QueryClient();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ConfigProvider
          locale={zhCN}
          theme={{
            algorithm: theme.defaultAlgorithm,
            token: {
              colorPrimary: "#124aa5",
              colorInfo: "#124aa5",
              colorBgContainer: "#ffffff",
              colorBorderSecondary: "rgba(15, 23, 42, 0.12)",
              borderRadius: 10,
              controlHeight: 38,
              fontFamily: "'IBM Plex Sans', 'Noto Sans SC', 'Source Han Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif",
            },
            components: {
              Button: {
                fontWeight: 500,
                controlHeightLG: 42,
              },
              Card: {
                headerHeight: 54,
              },
            },
          }}
        >
          <App />
        </ConfigProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>
);
