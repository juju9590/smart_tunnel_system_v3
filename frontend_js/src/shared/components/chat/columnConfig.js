export const COLUMN_MAP = {
  // 공통
  id: 'ID',
  // detection_results
  event_type: '이벤트 유형',
  address: '주소',
  latitude: '위도',
  longitude: '경도',
  detected_at: '감지 시각',
  is_simulation: '시뮬레이션 여부',
  video_origin: '영상 출처',
  is_resolved: '처리 완료',
  resolved_at: '처리 시각',
  resolved_by: '처리자',
  feedback: '피드백',
  // plate_results
  plate_number: '번호판',
  ground_truth: '실제 번호판',
  is_correct: '인식 정확도',
  confidence: '신뢰도',
  vote_count: '투표 수',
  is_fixed: '수정 여부',
  img_path: '이미지',       // 🟢 수정: '이미지 경로' → '이미지'로 통일
  video_filename: '영상 파일명',
  operator_name: '담당자',
  created_at: '생성 시각',
  updated_at: '수정 시각',
  // 사진 모드 SQL 별칭 (대소문자 모두 대응)
  IMAGE_PATH: '이미지',     // 🟢 수정: 사진 모드 AS IMAGE_PATH 대응
  image_path: '이미지',     // 🟢 수정: 소문자도 대응
  // UNION ALL 결과
  type: '유형',
  category: '상세',
};
 
export const formatValue = (key, val) => {
  if (val === null || val === undefined) return '-';
  if (key === 'is_simulation') return val == 1 ? '🟡 시뮬레이션' : '🔴 실제상황';
  if (key === 'is_resolved') return val == 1 ? '✅ 완료' : '⏳ 미완료';
  if (key === 'is_correct' || key === 'is_fixed') return val == 1 ? '✅' : '❌';
 
  // 날짜 포맷
  if (['detected_at', 'resolved_at', 'created_at', 'updated_at'].includes(key)) {
    const d = new Date(val);
    if (!isNaN(d)) {
      const pad = n => String(n).padStart(2, '0');
      return `${d.getFullYear()}년 ${pad(d.getMonth()+1)}월 ${pad(d.getDate())}일 ${pad(d.getHours())}시 ${pad(d.getMinutes())}분 ${pad(d.getSeconds())}초`;
    }
  }
 
  return String(val);
};