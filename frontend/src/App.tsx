import { BarChartOutlined, FileSearchOutlined, SafetyCertificateOutlined } from "@ant-design/icons";
import { Layout, Menu, Typography } from "antd";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";

import { AnalysisPage } from "./pages/AnalysisPage";
import { BatchTasksPage } from "./pages/BatchTasksPage";
import { ReviewPage } from "./pages/ReviewPage";

const { Header, Content } = Layout;

const menuItems = [
  { key: "/analysis", icon: <SafetyCertificateOutlined />, label: "首页分析" },
  { key: "/batch", icon: <FileSearchOutlined />, label: "批量任务" },
  { key: "/review", icon: <BarChartOutlined />, label: "人工复核" },
];

function App() {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <Layout className="app-shell">
      <Header className="app-header">
        <div className="app-brand">
          <Typography.Title level={4} className="app-brand-title">
            金融监管问答智能工作台
          </Typography.Title>
          <Typography.Text className="app-brand-subtitle">
            专业评估 · 高效复核 · 低动效轻体验
          </Typography.Text>
        </div>
        <Menu
          mode="horizontal"
          theme="dark"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={(event) => navigate(event.key)}
          style={{ minWidth: 360, justifyContent: "flex-end", borderBottom: "none", background: "transparent" }}
        />
      </Header>
      <Content className="app-content">
        <Routes>
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/batch" element={<BatchTasksPage />} />
          <Route path="/review" element={<ReviewPage />} />
          <Route path="*" element={<Navigate to="/analysis" replace />} />
        </Routes>
      </Content>
    </Layout>
  );
}

export default App;
