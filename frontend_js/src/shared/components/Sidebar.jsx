/* eslint-disable */
import React, { useState } from 'react';

function Sidebar({ activeTab, onTabChange, user, onLogout, isMobile }) {
  const [isCollapsed, setIsCollapsed] = useState(true);

  const menuItems = [
    { id: "cctv",       label: "CCTV 모니터링",      icon: "📡" },
    { id: "monitoring", label: "교통 흐름 모니터링",   icon: "🚦" },
    {
      id: "tunnel", label: "스마트 터널 모니터링",
      icon: (
        <img
          src="/tunnel.jpg"
          alt="Tunnel Icon"
          style={{ width: '18px', height: '18px', verticalAlign: 'middle' }}
        />
      ),
    },
    { id: "raspi",  label: "라즈베리파이 CCTV", icon: "🖥️" },
    { id: "plate",  label: "번호판 인식",        icon: "🔍" },
    { id: "stats",  label: "통계 데이터",        icon: "📊" },
  ];

  const sidebarWidth = isMobile ? '100%' : (isCollapsed ? '80px' : '240px');

  // ── 모바일 레이아웃 ──────────────────────────────────────────────────
  if (isMobile) {
    return (
      <aside style={mobileAside}>
        {/* 상단: 로고 + 로그아웃 버튼 */}
        <div style={mobileHeader}>
          <div style={mobileLogoRow}>
            <div style={mobileLogoIcon}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"
                style={{ width: '16px', color: '#6366f1' }}>
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
            </div>
            <span style={mobileLogoText}>TADS</span>
          </div>

          {/* ✅ 항상 보이는 로그아웃 버튼 */}
          <button onClick={onLogout} style={mobileLogoutBtn}>
            Exit
          </button>
        </div>

        {/* 탭 스크롤 영역 */}
        <nav style={mobileNav}>
          {menuItems.map((menu) => {
            const isActive = activeTab === menu.id;
            return (
              <div
                key={menu.id}
                onClick={() => onTabChange(menu.id)}
                style={{
                  ...mobileTab,
                  color:           isActive ? '#818cf8' : '#64748b',
                  backgroundColor: isActive ? 'rgba(99,102,241,0.12)' : 'transparent',
                  borderBottom:    isActive ? '2px solid #6366f1' : '2px solid transparent',
                }}
              >
                <span style={{ fontSize: '20px', lineHeight: 1 }}>{menu.icon}</span>
                <span style={mobileTabLabel}>{menu.label}</span>
              </div>
            );
          })}
        </nav>
      </aside>
    );
  }

  // ── 데스크톱 레이아웃 (기존 유지) ────────────────────────────────────
  return (
    <aside style={{
      ...sidebarStyle,
      width: sidebarWidth,
      minWidth: sidebarWidth,
      height: '100vh',
      transition: 'width 0.35s cubic-bezier(0.4, 0, 0.2, 1)',
      position: 'relative',
    }}>
      <button onClick={() => setIsCollapsed(!isCollapsed)} style={toggleButtonStyle}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
          style={{ transform: isCollapsed ? 'rotate(0deg)' : 'rotate(180deg)', transition: 'transform 0.3s' }}>
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </button>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
        <div style={{ ...logoArea, padding: '30px 20px', justifyContent: isCollapsed ? 'center' : 'flex-start' }}>
          <div style={{ ...logoIcon, width: '40px', height: '40px', minWidth: '40px' }}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"
              style={{ width: '24px', color: '#6366f1' }}>
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </div>
          {!isCollapsed && (
            <div style={{ marginLeft: '12px', animation: 'fadeIn 0.3s' }}>
              <div style={{ ...logoTitle, fontSize: '22px' }}>TADS</div>
              <div style={{ ...logoSub, fontSize: '11px' }}>관제 센터 시스템</div>
            </div>
          )}
        </div>

        <nav style={{ ...sideNavStyle, flexDirection: 'column', padding: '10px 0' }}>
          {!isCollapsed && (
            <div style={{ ...menuLabel, fontSize: '12px', marginBottom: '15px' }}>메인 메뉴</div>
          )}
          {menuItems.map((menu) => (
            <div
              key={menu.id}
              onClick={() => onTabChange(menu.id)}
              title={isCollapsed ? menu.label : ''}
              style={{
                ...menuItemStyle,
                padding: '18px 0',
                justifyContent: isCollapsed ? 'center' : 'flex-start',
                paddingLeft: isCollapsed ? '0' : '25px',
                backgroundColor: activeTab === menu.id ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
                color: activeTab === menu.id ? '#818cf8' : '#94a3b8',
                borderLeft: (!isCollapsed && activeTab === menu.id) ? '4px solid #6366f1' : 'none',
              }}
            >
              <span style={{
                marginRight: isCollapsed ? '0' : '12px',
                fontSize: '20px',
                width: isCollapsed ? '100%' : 'auto',
                textAlign: 'center',
              }}>
                {menu.icon}
              </span>
              {!isCollapsed && (
                <span style={{
                  fontWeight: activeTab === menu.id ? '700' : '500',
                  fontSize: '15px',
                  whiteSpace: 'nowrap',
                }}>
                  {menu.label}
                </span>
              )}
            </div>
          ))}
        </nav>
      </div>

      <div style={{
        ...sidebarFooter,
        textAlign: isCollapsed ? 'center' : 'left',
        padding: isCollapsed ? '20px 10px' : '20px 25px',
      }}>
        {isCollapsed ? (
          <div style={{ ...statusDot, margin: '0 auto 15px auto' }} title={`${user?.name} 관리자님 접속 중`} />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '15px' }}>
            <div style={{ fontSize: '12px', color: '#818cf8', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ fontSize: '14px' }}>👤</span> {user?.name} 관리자
            </div>
            <div style={statusWrapper}>
              <div style={statusDot} />
              <span style={statusText}>시스템 온라인</span>
            </div>
          </div>
        )}
        <button
          onClick={onLogout}
          style={{
            ...logoutBtn,
            fontSize: isCollapsed ? '10px' : '13px',
            padding: isCollapsed ? '8px 0' : '10px',
            marginTop: isCollapsed ? '0' : '5px',
          }}
        >
          {isCollapsed ? 'Exit' : '로그아웃'}
        </button>
      </div>
    </aside>
  );
}

// ── 모바일 전용 스타일 ──────────────────────────────────────────────────
const mobileAside = {
  width: '100%',
  background: '#0f172a',
  borderBottom: '1px solid #1e293b',
  zIndex: 100,
  flexShrink: 0, // ✅ 세로 레이아웃에서 찌그러지지 않도록
};

const mobileHeader = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '10px 14px 6px',
};

const mobileLogoRow = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
};

const mobileLogoIcon = {
  background: '#1e293b',
  borderRadius: '8px',
  width: '28px', height: '28px',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
};

const mobileLogoText = {
  fontWeight: '900',
  fontSize: '18px',
  color: '#fff',
  letterSpacing: '1px',
};

// ✅ 항상 노출되는 로그아웃 버튼
const mobileLogoutBtn = {
  background: '#1e293b',
  border: '1px solid #334155',
  color: '#94a3b8',
  borderRadius: '8px',
  cursor: 'pointer',
  fontSize: '12px',
  padding: '6px 14px',
  transition: 'all 0.2s',
};

// ✅ 가로 스크롤 탭바
const mobileNav = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  flexDirection: 'row',
  overflowX: 'auto',            // 탭 넘칠 때 가로 스크롤
  WebkitOverflowScrolling: 'touch',
  padding: '0 10px',
  gap: '2px',
  // 스크롤바 숨기기
  msOverflowStyle: 'none',
  scrollbarWidth: 'none',
};

const mobileTab = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '3px',
  padding: '8px 10px',
  cursor: 'pointer',
  flexShrink: 0,                // ✅ 탭이 찌그러지지 않도록
  minWidth: '60px',
  borderRadius: '6px 6px 0 0',
  transition: 'all 0.15s',
  userSelect: 'none',
};

const mobileTabLabel = {
  fontSize: '9px',
  whiteSpace: 'nowrap',
  fontWeight: '500',
  lineHeight: 1.2,
  textAlign: 'center',
};

// ── 데스크톱 전용 스타일 (기존 유지) ──────────────────────────────────
const sidebarStyle      = { background: '#0f172a', borderRight: '1px solid #1e293b', display: 'flex', flexDirection: 'column', zIndex: 100 };
const logoArea          = { display: 'flex', alignItems: 'center', overflow: 'hidden' };
const logoIcon          = { background: '#1e293b', borderRadius: '12px', display: 'flex', justifyContent: 'center', alignItems: 'center', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' };
const logoTitle         = { fontWeight: '900', color: '#fff', letterSpacing: '1px' };
const logoSub           = { color: '#64748b', whiteSpace: 'nowrap' };
const sideNavStyle      = { display: 'flex' };
const menuLabel         = { color: '#475569', padding: '0 25px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' };
const menuItemStyle     = { cursor: 'pointer', display: 'flex', alignItems: 'center', transition: 'all 0.2s ease' };
const sidebarFooter     = { borderTop: '1px solid #1e293b' };
const statusWrapper     = { display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' };
const statusDot         = { width: '8px', height: '8px', background: '#22c55e', borderRadius: '50%', boxShadow: '0 0 8px #22c55e' };
const statusText        = { fontSize: '11px', color: '#94a3b8', fontWeight: '500' };
const logoutBtn         = { width: '100%', background: '#1e293b', border: '1px solid #334155', color: '#94a3b8', borderRadius: '8px', cursor: 'pointer', transition: 'all 0.2s' };
const toggleButtonStyle = { position: 'absolute', right: '-12px', top: '40px', width: '26px', height: '26px', borderRadius: '50%', background: '#6366f1', color: 'white', border: '2px solid #0f172a', cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 101, boxShadow: '0 4px 10px rgba(0,0,0,0.3)', padding: 0 };

export default Sidebar;