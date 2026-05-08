import React from "react";
import dashboardPreview from "../../assets/landing-dashboard.png";
import landingFeatures from "../../assets/landing-features.png";
import landingMetrics from "../../assets/landing-metrics.png";
import landingWorkflow from "../../assets/landing-workflow.png";
import tunnelLogo from "../../assets/tunnel-logo.png";

function openExternalUrl(url) {
  if (!url) return;
  window.open(url, "_blank", "noopener,noreferrer");
}

function TunnelLanding({ onStart, portfolioUrl, githubUrl }) {
  return (
    <div className="tunnel-landing">
      <header className="landing-nav">
        <div className="landing-brand">
          <img className="landing-brand-mark" src={tunnelLogo} alt="Tunnel AI logo" />
          <div>
            <strong>TUNNEL AI</strong>
            <em>스마트 터널 관제 시스템</em>
          </div>
        </div>

        <nav className="landing-nav-links" aria-label="landing navigation">
          <a href="#project-intro">프로젝트 소개</a>
          <a href="#landing-features">핵심 기능</a>
          <a href="#landing-metrics">성능 요약</a>
          <a href="#landing-workflow">시스템 흐름</a>
          <button type="button" onClick={() => openExternalUrl(githubUrl)}>
            GitHub
          </button>
        </nav>

        <button className="landing-nav-cta" type="button" onClick={onStart}>
          실시간 관제 시작
        </button>
      </header>

      <main className="landing-main">
        <section className="landing-hero" id="project-intro">
          <div className="landing-hero-content">
            <div className="landing-kicker">AI Tunnel Control Platform</div>
            <h1>
              <span>AI</span> Smart Tunnel <span>Monitoring</span> System
            </h1>
            <p>
              CCTV 영상만으로 차량 흐름, 사고 위험, 차선 상태, 환기 대응을
              실시간 분석하는 AI 기반 스마트 터널 관제 시스템입니다.
            </p>

            <div className="landing-hero-badges">
              <span>Real-time CCTV</span>
              <span>AI Incident Detection</span>
              <span>Sensorless Ventilation</span>
            </div>

            <div className="landing-actions">
              <button className="landing-button primary" type="button" onClick={onStart}>
                <span className="landing-button-icon">LIVE</span>
                실시간 관제 시작
              </button>
              <button
                className="landing-button secondary"
                type="button"
                onClick={() => openExternalUrl(portfolioUrl)}
              >
                <span className="landing-button-icon">PF</span>
                포트폴리오 보기
              </button>
              <button
                className="landing-button ghost"
                type="button"
                onClick={() => openExternalUrl(githubUrl)}
              >
                <span className="landing-button-icon">GH</span>
                GitHub 보기
              </button>
            </div>
          </div>

          <div className="landing-preview-card">
            <div className="landing-preview-topbar">
              <div>
                <span className="landing-status-dot" />
                <strong>Smart Tunnel Dashboard</strong>
              </div>
              <span>25 FPS</span>
            </div>

            <div className="landing-preview-image-wrap">
              <img
                className="landing-preview-image"
                src={dashboardPreview}
                alt="Smart Tunnel dashboard preview"
              />
            </div>

            <div className="landing-preview-caption">
              <span>LIVE AI ANALYSIS</span>
              <strong>CCTV 기반 실시간 관제 대시보드</strong>
            </div>
          </div>
        </section>

        <section className="landing-section landing-image-section" aria-labelledby="landing-feature-title">
          <div className="landing-card landing-image-card landing-feature-panel" id="landing-features">
            <div className="landing-section-head">
              <span>Pipeline</span>
              <strong id="landing-feature-title">핵심 기능</strong>
            </div>
            <img
              className="landing-section-image"
              src={landingFeatures}
              alt="핵심 기능 섹션"
            />
          </div>

          <div className="landing-card landing-image-card landing-metric-panel" id="landing-metrics">
            <div className="landing-section-head">
              <span>Validation</span>
              <strong>성능 요약</strong>
            </div>
            <img
              className="landing-section-image"
              src={landingMetrics}
              alt="성능 요약 섹션"
            />
          </div>
        </section>

        <section className="landing-flow-section landing-card landing-image-card" id="landing-workflow">
          <div className="landing-section-head">
            <span>Workflow</span>
            <strong>시스템 흐름</strong>
          </div>
          <img
            className="landing-section-image landing-workflow-image"
            src={landingWorkflow}
            alt="시스템 흐름 섹션"
          />
        </section>
      </main>
    </div>
  );
}

export default TunnelLanding;
