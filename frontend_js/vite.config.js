import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig(({ mode }) => {
  // vitest 설정: jsdom 환경에서 React 컴포넌트·훅 테스트 실행
  // `npm test` 또는 `npx vitest run` 으로 실행
  const test = {
    environment: 'jsdom',        // 브라우저 DOM API 시뮬레이션
    globals: true,               // describe/it/expect 전역 사용 가능
    setupFiles: [],              // 전역 setup 파일 (필요 시 추가)
  };
  const rootPath = path.resolve(__dirname, '../');
  const env = loadEnv(mode, rootPath, '');

  const kakaoKey = env.VITE_KAKAO_MAP_API_KEY 
                || process.env.VITE_KAKAO_MAP_API_KEY 
                || "";

  return {
    server: {
      fs: { allow: [rootPath] },
      host: '0.0.0.0',
      port: 5173,
      cors: true,
      allowedHosts: ['.ts.net']
    },
    plugins: [react()],
    define: {
      __KAKAO_KEY__: JSON.stringify(kakaoKey)  // 빌드 시 번들에 값 직접 삽입
    },
    test,  // vitest 설정 주입
  }
})