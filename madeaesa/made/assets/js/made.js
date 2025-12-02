/**
 * 감정 분석 렌더러 및 헬퍼 함수들
 * analysis.js의 모든 기능을 통합
 */

// 전역 변수
const MAIN_LABELS = { '1': '희', '2': '노', '3': '애', '4': '락' };
const DEFAULT_SUB_LABEL = { '희': '감사', '노': '분노', '애': '슬픔', '락': '안심' };
const DEFAULT_SAMPLE_INPUT = '병원 카운터 직원에게 어제 구매한 연고를 반품해 달라고 했더니 “몇천원 밖에 안하는데 그냥 쓰시죠?”라는 말을 들었습니다. 그 답변이 무시당한 느낌을 줘서 기분이 많이 상했고, 다시는 그곳을 이용하고 싶지 않을 정도로 분노가 치밀었습니다.';
const MODULE_STATUS_LABEL = { ok: 'OK', missing: '데이터 없음', skipped: 'SKIP', error: 'ERROR' };
const MODULE_STATUS_COLORS = { ok: '#10b981', missing: '#f87171', skipped: '#fbbf24', error: '#fb7185' };
const MODULE_DISPLAY_INFO = {
    linguistic_matcher: { label: 'linguistic_matcher (언어 패턴 매칭)', desc: '감정 키워드 및 구문 패턴 분석' },
    pattern_extractor: { label: 'pattern_extractor (패턴 추출)', desc: '문장 내 반복 패턴 및 템플릿 분석' },
    context_analysis: { label: 'context_analysis (맥락 해석)', desc: '컨텍스트 스코어 및 트리거 기반 판단' },
    context_extractor: { label: 'context_extractor (컨텍스트 추출)', desc: '엔티티·키워드 기반 맥락 추출' },
    time_series_analyzer: { label: 'time_series_analyzer (시계열 분석)', desc: '감정 변화 시간적 패턴' },
    transition_analyzer: { label: 'transition_analyzer (전이 분석)', desc: '감정 전환 패턴 추출' },
    relationship_analyzer: { label: 'relationship_analyzer (감정 관계 분석)', desc: '문장별 감정 유기 구조 분석' },
    situation_analyzer: { label: 'situation_analyzer (상황 분석)', desc: '상황별 감정 매칭' },
    intensity_analyzer: { label: 'intensity_analyzer (강도 분석)', desc: '감정 세기 및 신뢰도 측정' },
    psychological_analyzer: { label: 'psychological_analyzer (심리 분석)', desc: '심리/인지 패턴 탐지' },
    complex_analyzer: { label: 'complex_analyzer (복합 분석)', desc: '다층적 감정 조합 및 상호작용 분석' },
    weight_calculator: { label: 'weight_calculator (가중치 계산)', desc: '감정 강도·특징 중요도 산출' },
};
const MODULE_RESULT_FALLBACK = { relationship_analyzer: 'emotion_relationship_analyzer' };
function resolveApiUrl(path) {
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    if (window.API_BASE_URL) {
        const base = window.API_BASE_URL.replace(/\/+$/, '');
        return `${base}${normalizedPath}`;
    }
    // 기본값: 현재 페이지와 동일한 origin(프로토콜 + host)을 사용
    const { protocol, host } = window.location; // host = hostname[:port]
    if (host) {
        return `${protocol}//${host}${normalizedPath}`;
    }
    // 최후 수단: 상대 경로로 요청 (동일 origin)
    return normalizedPath;
}

function getMasterSection(masterReport, title) {
    if (typeof masterReport !== 'string' || !title) return [];
    const lines = masterReport.split('\n');
    const header = `== ${title} ==`;
    let start = -1;
    for (let i = 0; i < lines.length; i += 1) {
        if (lines[i].trim() === header) {
            start = i + 1;
            break;
        }
    }
    if (start === -1) return [];

    const section = [];
    for (let i = start; i < lines.length; i += 1) {
        const raw = lines[i];
        const trimmed = raw.trim();
        if (trimmed.startsWith('==') && trimmed.endsWith('==') && trimmed.length > 4) {
            break;
        }
        if (!trimmed || /^=+$/.test(trimmed)) {
            continue;
        }
        if (trimmed.includes('| INFO |') || trimmed.startsWith('[Kss]')) {
            continue;
        }
        section.push(raw.replace(/\s+$/g, ''));
    }
    return section;
}

function chooseLines(primary, fallback) {
    const primaryLines = Array.isArray(primary) ? primary.filter(line => typeof line === 'string' && line.trim()) : [];
    if (primaryLines.length) return primaryLines;
    return Array.isArray(fallback) ? fallback.filter(line => typeof line === 'string' && line.trim()) : [];
}

// 헬퍼 함수들
function formatEmotions(emotions) {
    if (!emotions || emotions.length === 0) return '—';
    if (typeof emotions[0] === 'object' && emotions[0].name) {
        return emotions.map(e => `${e.name} ${e.pct}%`).join(', ');
    }
    return emotions.join(', ');
}

function formatArray(arr) {
    if (!arr || arr.length === 0) return '—';
    return arr.join(', ');
}

function getMaturityLevel(maturity) {
    if (!maturity) return '낮음';
    if (maturity >= 80) return '높음';
    if (maturity >= 50) return '중간';
    return '낮음';
}

function truncateText(text, maxLength) {
    if (!text) return '';
    const str = String(text);
    return str.length > maxLength ? `${str.slice(0, maxLength - 1)}…` : str;
}

function mapMainLabel(value) {
    if (value === undefined || value === null) return '—';
    const key = String(value).trim();
    return MAIN_LABELS[key] || key;
}

function mapSubLabel(main, raw, aliasMap) {
    if (!raw) return '';
    const str = String(raw).trim();
    if (!str) return '';
    if (aliasMap && (aliasMap[str] || aliasMap[`${main}-${str}`])) {
        return aliasMap[str] || aliasMap[`${main}-${str}`];
    }
    if (str.includes('-') && !str.includes('sub_')) {
        const parts = str.split('-');
        return parts[parts.length - 1];
    }
    // sub_ 형식이 남아있으면 메인 감정에 따른 기본값 반환
    if (str.includes('sub_')) {
        return DEFAULT_SUB_LABEL[main] || DEFAULT_SUB_LABEL[mapMainLabel(main)] || '중립';
    }
    return str;
}

function normalizeSubLabel(main, sub) {
    const mainLabel = mapMainLabel(main);
    const raw = typeof sub === 'string' ? sub.trim() : (sub ? String(sub).trim() : '');
    if (!raw || raw === '—') {
        return DEFAULT_SUB_LABEL[mainLabel] || '—';
    }
    return raw;
}

// 분석 렌더러
const AnalysisRenderer = {
    getEl(id) {
        return document.getElementById(id);
    },

    reset(message = '') {
        const slots = [
            'analysisExecutionSummary',
            'analysisKpiCards',
            'analysisRiskActions',
            'analysisEmotionDist',
            'analysisTransitions',
            'analysisSentenceAnnotations',
            'analysisProductSpecs',
            'analysisExplainability',
            'analysisModelNarrative',
            'analysisInsightSummary',
            'analysisInvestorHighlights',
            'analysisStrategicBrief',
            'analysisMasterReport'
        ];

        slots.forEach((id) => {
            const el = this.getEl(id);
            if (el) {
                el.innerHTML = '';
            }
        });

        this.showError('');
        if (message) {
            this.setStatus(message);
        } else {
            this.setStatus('');
        }
    },

    setStatus(message, isError = false) {
        const status = this.getEl('analysisStatus');
        if (status) {
            status.textContent = message;
            status.className = isError ? 'analysis-refine-status error' : 'analysis-refine-status';
        }
    },

    showError(message = '') {
        const errorEl = this.getEl('analysisError');
        if (errorEl) {
            errorEl.style.display = message ? 'block' : 'none';
            errorEl.textContent = message;
        }
    },

    render(rawData) {
        if (!rawData) return;

        const data = alignResultData(rawData);
        this.showError();
        
        // 실행 요약
        this.renderExecutionSummary(data);
        
        // 핵심 KPI
        this.renderKpiCards(data);
        
        // 리스크 & 권장 액션
        this.renderRiskActions(data);
        
        // 감정 분포 (세부 감정 포함)
        // ★★★ Truth 필드 우선 사용: data.truth.main_dist (test.py 원본 보존) ★★★
        // poster.main_distribution은 문장 기반 재계산으로 덮어씌워질 수 있으므로 사용하지 않음
        const truth = data.truth || {};
        const bundle = data.bundle || {};
        const mainDist = truth.main_dist || bundle.main_dist || (data.poster && data.poster.main_distribution) || (data.main_distribution || {});
        this.renderEmotionDistribution(mainDist, data);
        
        // 감정 전이
        this.renderTransitions(data);
        
        // 문장 주석(감정 태깅)
        this.renderSentenceAnnotations(data);
        
        // 제품 스펙
        this.renderProductSpecs(data);
        
        // 설명 가능성
        this.renderExplainability(data);
        this.renderModelNarrative(data);
        
        // 인사이트 요약
        this.renderInsightSummary(data);
        
        // 투자 하이라이트
        this.renderInvestorHighlights(data);
        
        // 전략 브리프
        this.renderStrategicBrief(data);
        
        // 마스터 리포트
        this.renderMasterReport(data);

        // Expert View (전문가 모드) - Truth 필드 원본 표시
        this.renderExpertView(data);

        const modeLabel = (data.meta && typeof data.meta.mode === 'string')
            ? data.meta.mode.toUpperCase()
            : (typeof data.mode === 'string' ? data.mode.toUpperCase() : 'BALANCED');
        this.setStatus(`${modeLabel} 모드 분석 완료`);
    },

    renderExecutionSummary(data) {
        const container = this.getEl('analysisExecutionSummary');
        if (!container) return;

        const meta = data.meta || {};
        const items = [];

        // 분석한 텍스트를 가장 위에 표시
        const analyzedText = data.text || data.input_text || data.inputText || '';
        if (analyzedText) {
            const textDisplay = analyzedText.length > 200 
                ? analyzedText.substring(0, 200) + '...' 
                : analyzedText;
            items.push(`<div class="analyzed-text" style="margin-bottom: 1rem; padding: 0.75rem; background: rgba(16,185,129,0.1); border-left: 3px solid #10b981; border-radius: 4px; color: #e5e5e5; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word;">${textDisplay.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>`);
        }

        const elapsed = Number(meta.elapsed ?? 0);
        const evidence = meta.evidence_score;
        
        if (!Number.isNaN(elapsed) && elapsed > 0) {
            items.push(`<span class="meta-item"> elapsed: ${elapsed.toFixed(3)}s</span>`);
        }
        if (meta.mode) {
            items.push(`<span class="meta-item">📊 mode: ${meta.mode}</span>`);
        }
        if (typeof evidence === 'number' && !Number.isNaN(evidence)) {
            items.push(`<span class="meta-item"> evidence: ${evidence.toFixed(2)}</span>`);
        }
        if (data.timestamp) {
            items.push(`<span class="meta-item"> timestamp: ${data.timestamp}</span>`);
        }
        if (meta.refined) {
            items.push('<span class="meta-item"> refined</span>');
        }

        const moduleDetails = Array.isArray(data.module_details) ? data.module_details : [];
        if (moduleDetails.length) {
            const okCount = moduleDetails.filter(detail => detail.status === 'ok').length;
            const total = moduleDetails.length;
            const conciseLine = moduleDetails
                .map(detail => {
                    const label = detail.name.replace(/_analyzer$/, '').replace(/_/g, ' ');
                    const status = detail.status === 'ok' ? 'OK' : detail.status.toUpperCase();
                    return `${label}:${status}`;
                })
                .join(' · ');
            items.push(`<span class="meta-item module-mini">Modules ${okCount}/${total} OK · ${conciseLine}</span>`);
        } else if (data.module_hit_rate) {
            const map = data.module_hit_rate;
            const entries = Object.keys(map || {});
            if (entries.length) {
                const okCount = entries.filter(key => map[key]).length;
                const total = entries.length;
                items.push(`<span class="meta-item module-mini">Modules ${okCount}/${total} OK</span>`);
            }
        }

        container.innerHTML = items.length ? items.join(' ') : '<span class="meta-item">결과 정보가 없습니다.</span>';
    },

    /**
     * Business View: 핵심 KPI 카드 렌더링
     * 
     * Truth 필드 연결:
     * - 주요 감정: data.truth.main_dist (메인 감정 분포) 또는 data.bundle.products.p1.headline_emotions
     * - Churn 위험: data.truth.products.p1.churn_probability
     * - 증거 점수: data.truth.meta.evidence_score
     * - 심리 안정성: data.truth.products.p5.stability
     * - 실행 모드: data.truth.meta.mode
     */
    renderKpiCards(data) {
        const container = this.getEl('analysisKpiCards');
        if (!container) return;

        // Truth 필드 우선 사용 (있으면), 없으면 기존 필드 사용 (하위 호환성)
        const truth = data.truth || {};
        const bundle = data.bundle || {};
        
        const meta = truth.meta || data.meta || {};
        const poster = data.poster || {};
        const domainProfile = (meta.domain_profile || poster.domain_profile || '').toLowerCase();
        const isGeneric = domainProfile && domainProfile !== 'service';

        // Truth 필드에서 products 가져오기 (우선순위: truth.products > bundle.products > data.products)
        const truthProducts = truth.products || bundle.products || {};
        const p1 = truthProducts.p1 || data.products?.p1 || {};
        const p5 = truthProducts.p5 || data.products?.p5 || {};
        const trust = poster.trust_stamp || {};
        const genericHighlights = Array.isArray(truthProducts.generic?.highlights || data.products?.generic?.highlights)
            ? (truthProducts.generic?.highlights || data.products.generic.highlights)
            : [];

        // Truth 필드에서 main_dist 가져오기
        const mainDist = truth.main_dist || bundle.main_dist || poster.main_distribution || {};
        
        // 디버깅: mainDist 확인
        console.log('[renderKpiCards] mainDist 확인:', {
            'truth.main_dist': truth.main_dist,
            'bundle.main_dist': bundle.main_dist,
            'poster.main_distribution': poster.main_distribution,
            '최종 mainDist': mainDist,
            'p1.headline_emotions': p1.headline_emotions
        });
        
        // ★★★ mainDist가 있으면 직접 사용하여 headline_emotions 생성 ★★★
        let headlineEmotionsFromMainDist = null;
        if (mainDist && typeof mainDist === 'object' && Object.keys(mainDist).length > 0) {
            // mainDist에서 상위 3개 감정 추출 (0이 아닌 값만)
            const sortedEmotions = Object.entries(mainDist)
                .filter(([key, value]) => value > 0)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .map(([name, value]) => ({
                    name: name,
                    pct: Number((value * 100).toFixed(1))
                }));
            
            if (sortedEmotions.length > 0) {
                headlineEmotionsFromMainDist = sortedEmotions;
                console.log('[renderKpiCards] ✅ mainDist에서 headline_emotions 생성:', headlineEmotionsFromMainDist);
            }
        }
        
        const fallbackHeadline = (poster.main && mainDist[poster.main])
            ? [{ name: poster.main, pct: Number(((mainDist[poster.main] || 0) * 100).toFixed(1)) }]
            : [];

        // 우선순위: p1.headline_emotions > mainDist 기반 생성 > fallbackHeadline > '—'
        const mainEmotionValue = formatEmotions(headlineEmotionsFromMainDist)
            || formatEmotions(p1.headline_emotions)
            || formatEmotions(fallbackHeadline)
            || '—';

        const churnValue = typeof p1.churn_probability === 'number' && !isGeneric
            ? `${p1.churn_probability}%`
            : '—';
        const churnSub = !isGeneric && typeof p1.horizon_days === 'number'
            ? `예상 기간: ${p1.horizon_days}일`
            : (isGeneric ? '일반 감성 분석 모드에서는 비활성화된 지표입니다.' : '');
        const churnHint = isGeneric
            ? '컨택센터/정책 맥락이 아니므로 이탈 위험을 추정하지 않았습니다.'
            : '3일 이내 이탈 가능성을 예측한 지표 (근거: data.truth.products.p1)';

        const cards = [
            {
                title: '주요 감정',
                value: mainEmotionValue,
                sub: p1.intensity ? `강도: ${p1.intensity}` : '',
                hint: '텍스트 전반에서 가장 높은 비중을 차지한 감정 (근거: data.truth.main_dist)',
                keep: true,
                source: 'data.truth.main_dist'
            },
            {
                title: 'Churn 위험',
                value: churnValue,
                sub: churnSub,
                hint: churnHint,
                keep: isGeneric,
                source: 'data.truth.products.p1'
            },
            {
                title: '증거 점수',
                value: typeof meta.evidence_score === 'number'
                    ? meta.evidence_score.toFixed(2)
                    : (trust.evidence || '—'),
                sub: trust.consistency !== undefined ? `일관성: ${trust.consistency}%` : '',
                hint: '인사이트의 신뢰도(1.0에 가까울수록 확실) (근거: data.truth.meta.evidence_score)',
                source: 'data.truth.meta.evidence_score'
            },
            {
                title: '심리 안정성',
                value: typeof p5.stability === 'number' ? `${p5.stability}%` : '—',
                sub: typeof p5.maturity === 'number' ? `감정 성숙도: ${p5.maturity}%` : '',
                hint: '감정 기복과 부정 신호를 토대로 산출한 안정 지수 (근거: data.truth.products.p5)',
                source: 'data.truth.products.p5'
            },
            {
                title: '실행 모드',
                value: meta.mode || 'BALANCED',
                sub: typeof meta.elapsed === 'number' ? `Latency: ${meta.elapsed.toFixed(3)}s` : '',
                hint: '분석 모드와 처리 시간'
            }
        ];

        if (isGeneric && genericHighlights.length) {
            cards.push({
                title: '투자 인사이트',
                value: genericHighlights[0],
                sub: genericHighlights[1] || '',
                hint: '감정 여정에서 도출된 핵심 하이라이트',
                keep: true
            });
        }

        const visibleCards = cards.filter(card => card.keep || (card.value && card.value !== '—'));

        container.innerHTML = '';
        if (!visibleCards.length) {
            container.innerHTML = '<div class="empty-state">KPI 정보가 없습니다.</div>';
            return;
        }

        visibleCards.forEach(cardInfo => {
            const card = document.createElement('div');
            card.className = 'kpi-card';
            card.innerHTML = `
                <div class="kpi-title">${cardInfo.title}</div>
                <div class="kpi-value">${cardInfo.value}</div>
                ${cardInfo.sub ? `<div class="kpi-subtext">${cardInfo.sub}</div>` : ''}
                ${cardInfo.hint ? `<div class="kpi-subtext" style="color:#808080;">${cardInfo.hint}</div>` : ''}
            `;
            container.appendChild(card);
        });
    },

    /**
     * Business View: 리스크 & 권장 액션 렌더링
     * 
     * Truth 필드 연결:
     * - 이탈 위험: data.truth.products.p1.churn_probability
     * - 주요 트리거: data.truth.products.p1.triggers 또는 data.truth.triggers
     * - 권장 액션: data.truth.products.p1.recommended_actions
     * - 리스크 평가: data.truth.products.p3 (grade, risk_score, alert)
     */
    renderRiskActions(data) {
        const container = this.getEl('analysisRiskActions');
        if (!container) return;

        // Truth 필드 우선 사용
        const truth = data.truth || {};
        const bundle = data.bundle || {};
        const truthProducts = truth.products || bundle.products || {};
        
        const domainProfile = ((truth.meta || data.meta)?.domain_profile || data.poster?.domain_profile || '').toLowerCase();
        const isGeneric = domainProfile && domainProfile !== 'service';

        if (isGeneric) {
            const highlights = Array.isArray(truthProducts.generic?.highlights || data.products?.generic?.highlights)
                ? (truthProducts.generic?.highlights || data.products.generic.highlights)
                : [];
            container.innerHTML = '';
            if (!highlights.length) {
                container.innerHTML = '<div class="empty-state">감정 여정 하이라이트가 부족합니다.</div>';
                return;
            }
            const list = document.createElement('ul');
            highlights.forEach(item => {
                const li = document.createElement('li');
                li.textContent = item;
                list.appendChild(li);
            });
            container.appendChild(list);
            return;
        }

        // Truth 필드에서 products 가져오기
        const p1 = truthProducts.p1 || data.products?.p1 || {};
        const p3 = truthProducts.p3 || data.products?.p3 || {};
        const items = [];

        if (typeof p1.churn_probability === 'number') {
            const intensity = p1.intensity ? ` (강도: ${p1.intensity})` : '';
            const horizon = typeof p1.horizon_days === 'number' ? `${p1.horizon_days}일` : '3일';
            items.push(`${horizon} 내 이탈 위험: <strong>${p1.churn_probability}%</strong>${intensity}`);
        }

        if (Array.isArray(p1.triggers) && p1.triggers.length) {
            items.push(`주요 트리거: <strong>${p1.triggers.join(', ')}</strong>`);
        }

        if (Array.isArray(p1.recommended_actions) && p1.recommended_actions.length) {
            p1.recommended_actions.forEach(action => {
                items.push(`권장 액션: ${action}`);
            });
        }

        if (p3 && Object.keys(p3).length) {
            const grade = p3.grade ? `등급: ${p3.grade}` : null;
            const score = typeof p3.risk_score === 'number' ? `점수: ${p3.risk_score}` : null;
            const alert = p3.alert ? '⚠️ 경고 발생' : null;
            const summary = [grade, score, alert].filter(Boolean).join(' · ');
            if (summary) {
                items.push(`리스크 평가: ${summary}`);
            }
        }

        container.innerHTML = '';
        if (!items.length) {
            container.innerHTML = '<div class="empty-state">리스크 및 권장 액션 정보가 없습니다.</div>';
            return;
        }

        const list = document.createElement('ul');
        items.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = item;
            list.appendChild(li);
        });
        container.appendChild(list);
    },

    /**
     * Business View: 감정 분포 렌더링
     * 
     * Truth 필드 연결:
     * - 메인 감정 분포: data.truth.main_dist (희/노/애/락 → 0~1)
     * - 세부 감정: data.truth.sentence_annotations_structured (문장별 감정 태깅) 또는 data.truth.sub_top
     * - 감정 전이: data.truth.transitions_structured (감정 전이 구조)
     */
    renderEmotionDistribution(mainDist, data = {}) {
        const container = this.getEl('analysisEmotionDist');
        if (!container) return;
        
        container.innerHTML = '';
        
        // Truth 필드 우선 사용
        const truth = data.truth || {};
        const bundle = data.bundle || {};
        
        // Truth 필드에서 main_dist 가져오기 (우선순위: truth.main_dist > bundle.main_dist > mainDist 파라미터)
        const truthMainDist = truth.main_dist || bundle.main_dist || mainDist || {};
        
        // [개선] 고정된 4개 감정 루프 대신, 데이터에 있는 모든 키를 동적으로 렌더링
        // 터미널처럼 "부정계", "실망/불안" 등 복합 라벨도 표시 가능하도록 함
        const entries = Object.entries(truthMainDist)
            .map(([key, val]) => ({
                name: mapMainLabel(key), // 라벨 매핑 (필요시)
                rawName: key,
                value: Number(val)
            }))
            .filter(item => item.value > 0.001) // 0.1% 미만은 숨김 (노이즈 제거)
            .sort((a, b) => b.value - a.value); // 높은 순 정렬

        if (!entries.length) {
            container.innerHTML = '<div class="empty-state">감정 분포 데이터를 찾을 수 없습니다.</div>';
            return;
        }

        // 색상 매핑 (동적 라벨 대응)
        const getColor = (name) => {
            if (name.includes('희') || name.includes('긍정') || name.includes('기대') || name.includes('만족')) return '#b5cea8'; // 녹색 계열
            if (name.includes('락') || name.includes('즐거움') || name.includes('안심')) return '#d7ba7d'; // 노란색 계열
            if (name.includes('노') || name.includes('분노') || name.includes('불만')) return '#f48771'; // 붉은색 계열
            if (name.includes('애') || name.includes('슬픔') || name.includes('우울')) return '#9cdcfe'; // 파란색 계열
            if (name.includes('부정')) return '#f48771'; // 부정계 -> 붉은색
            return '#a1a1aa'; // 기본 회색
        };

        // Truth 필드에서 세부 감정 분포 가져오기
        const subTop = Array.isArray(truth.sub_top || bundle.sub_top || data.sub_top) 
            ? (truth.sub_top || bundle.sub_top || data.sub_top) 
            : [];
        
        // 세부 감정 매핑 (메인 감정별로 그룹화)
        const subEmotionsByMain = {};
        if (subTop.length > 0) {
            subTop.forEach(item => {
                const subLabel = (item.sub || item.name || item.label || '').trim();
                if (!subLabel || subLabel === '—' || subLabel === '-') return;
                
                // 해당 서브 감정이 속할 메인 감정 찾기 (가장 연관성 높은 것)
                // 여기서는 단순하게 현재 표시된 메인 감정 중 가장 높은 확률을 가진 것에 할당하거나,
                // 이름 매칭을 통해 할당 (예: '희-감사' -> '희')
                let targetMain = null;
                
                // 1. 이름에 힌트가 있는 경우 (희-감사)
                if (item.main) {
                    targetMain = mapMainLabel(item.main);
                } else {
                    // 2. 없는 경우 가장 높은 메인 감정에 할당 (단순화)
                    targetMain = entries[0].name; 
                }

                if (targetMain) {
                    if (!subEmotionsByMain[targetMain]) subEmotionsByMain[targetMain] = [];
                    // 중복 방지
                    if (!subEmotionsByMain[targetMain].some(s => s.label === subLabel)) {
                        subEmotionsByMain[targetMain].push({
                            label: subLabel,
                            score: Number(item.p || item.score || 0)
                        });
                    }
                }
            });
        }

        entries.forEach(entry => {
            const normalized = entry.value > 1 ? entry.value / 100 : entry.value;
            const percentage = (normalized * 100).toFixed(1);
            const color = getColor(entry.name);
            
            // 해당 메인 감정의 세부 감정 가져오기
            const subList = subEmotionsByMain[entry.name] || [];
            const subHtml = subList.length > 0 
                ? `<div class="emotion-sub" style="font-size: 0.75rem; color: #808080; margin-top: 2px; margin-left: 2px;">
                    ↳ ${subList.map(s => s.label).slice(0, 3).join(', ')}
                   </div>`
                : '';

            // 바 차트 스타일
            const barHtml = `
                <div class="emotion-bar-container" style="width: 100%; background: rgba(255,255,255,0.05); height: 6px; border-radius: 3px; margin-top: 6px; overflow: hidden;">
                    <div class="emotion-bar-fill" style="width: ${percentage}%; background: ${color}; height: 100%;"></div>
                </div>
            `;

            const item = document.createElement('div');
            item.className = 'emotion-item';
            // 기존 스타일 오버라이드 (더 넓게 사용)
            item.style.width = '100%';
            item.style.marginBottom = '12px';
            
            item.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="emotion-name" style="color: ${color}; font-weight: 600;">${entry.name}</div>
                    <div class="emotion-value" style="font-variant-numeric: tabular-nums;">${percentage}%</div>
                </div>
                ${barHtml}
                ${subHtml}
            `;
            container.appendChild(item);
        });
    },

    /**
     * Business View: 감정 전이 렌더링
     * 
     * Truth 필드 연결:
     * - 감정 전이 구조: data.truth.transitions_structured
     */
    renderTransitions(data) {
        const container = this.getEl('analysisTransitions');
        if (!container) return;

        // Truth 필드 우선 사용
        const truth = data.truth || {};
        const bundle = data.bundle || {};
        
        const normalizeTransition = (raw = {}) => {
            const fromMain = mapMainLabel(raw.from_main ?? raw.from ?? raw.src ?? raw.start);
            const toMain = mapMainLabel(raw.to_main ?? raw.to ?? raw.dest ?? raw.end);
            // 원본 sub 값을 그대로 사용 - normalizeSubLabel를 호출하지 않음
            const fromSub = raw.from_sub ?? raw.fromSub ?? DEFAULT_SUB_LABEL[fromMain] ?? '—';
            const toSub = raw.to_sub ?? raw.toSub ?? DEFAULT_SUB_LABEL[toMain] ?? '—';
            const fromIndex = raw.from_index ?? raw.fromIndex;
            const toIndex = raw.to_index ?? raw.toIndex;
            let probabilityPct = raw.probability_pct;
            if (typeof probabilityPct !== 'number' && typeof raw.probability === 'number') {
                const prob = Number(raw.probability);
                probabilityPct = prob <= 1 ? Math.round(prob * 100) : Math.round(prob);
            }
            if (typeof probabilityPct !== 'number' && typeof raw.confidence === 'number') {
                const conf = Number(raw.confidence);
                probabilityPct = conf <= 1 ? Math.round(conf * 100) : Math.round(conf);
            }
            return {
                from: fromMain,
                to: toMain,
                fromSub,
                toSub,
                span: fromIndex !== undefined || toIndex !== undefined
                    ? `${fromIndex ?? '?'} → ${toIndex ?? '?'}`
                    : '',
                probabilityPct,
                probabilityExplain: raw.probability_explain || raw.probabilityExplain || '',
                reason: raw.transition_reason || raw.transitionReason || '',
                trigger: raw.trigger || raw.pattern || '',
                excerptFrom: raw.excerpt_from || raw.from_sentence_text || raw.excerptFrom || '',
                excerptTo: raw.excerpt_to || raw.to_sentence_text || raw.excerptTo || '',
                type: raw.transition_type || raw.transitionType || ''
            };
        };

        let transitions = [];

        // Truth 필드에서 transitions_structured 가져오기 (우선순위: truth.transitions_structured > bundle.transitions_structured > data.transitions_structured)
        const structured = Array.isArray(truth.transitions_structured || bundle.transitions_structured || data.transitions_structured) 
            ? (truth.transitions_structured || bundle.transitions_structured || data.transitions_structured) 
            : [];
        if (structured.length) {
            transitions = structured.map(normalizeTransition);
        }

        if (!transitions.length) {
            const raw = data.results || {};
            const transitionModule = raw.transition_analyzer || {};
            if (Array.isArray(transitionModule.transitions) && transitionModule.transitions.length) {
                transitions = transitionModule.transitions.map(normalizeTransition);
            } else {
                const flowTransitions = raw.context_extractor?.flow_transitions;
                if (Array.isArray(flowTransitions) && flowTransitions.length) {
                    transitions = flowTransitions.map(normalizeTransition);
                }
            }
        }

        if (!transitions.length) {
            container.innerHTML = '<div class="empty-state">감정 전이 정보를 찾을 수 없습니다.</div>';
            return;
        }

        const hint = '<div style="color:#808080;margin-bottom:6px;">확률은 감정 분포와 문장 간 변화 신호를 조합한 추정치입니다.</div>';
        const body = transitions
            .map(t => {
                const metaPieces = [];
                if (typeof t.probabilityPct === 'number') {
                    const explain = t.probabilityExplain ? ` · ${t.probabilityExplain}` : '';
                    metaPieces.push(`확률: ${t.probabilityPct}%${explain}`);
                } else if (t.probabilityExplain) {
                    metaPieces.push(t.probabilityExplain);
                }
                if (t.trigger) {
                    metaPieces.push(`Trigger: ${truncateText(t.trigger, 40)}`);
                }
                if (t.span) {
                    metaPieces.push(`문장: ${t.span}`);
                }

                const excerpts = [];
                if (t.excerptFrom) {
                    excerpts.push(`<div class="explain-meta">From: ${truncateText(t.excerptFrom, 60)}</div>`);
                }
                if (t.excerptTo) {
                    excerpts.push(`<div class="explain-meta">To: ${truncateText(t.excerptTo, 60)}</div>`);
                }

                const reason = t.reason ? `<div class="transition-reason">${truncateText(t.reason, 120)}</div>` : '';
                const arrow = t.type === 'steady' ? '↔' : '→';
                const typeLabel = t.type === 'steady' ? '감정 유지' : (t.type === 'shift' ? '감정 전이' : '');

                return `
                    <div class="transition-item">
                        <strong>${typeLabel ? `${typeLabel} · ` : ''}${t.from}(${t.fromSub}) ${arrow} ${t.to}(${t.toSub})</strong>
                        ${metaPieces.length ? metaPieces.map(piece => `<span class="explain-meta">${piece}</span>`).join('') : ''}
                        ${reason}
                        ${excerpts.join('')}
                    </div>
                `;
            })
            .join('');
        container.innerHTML = hint + body;
    },

    /**
     * Business View: 문장 주석(감정 태깅) 렌더링
     * 
     * Truth 필드 연결:
     * - 문장별 감정 태깅: data.truth.sentence_annotations_structured
     */
    renderSentenceAnnotations(data) {
        const container = this.getEl('analysisSentenceAnnotations');
        if (!container) return;

        container.innerHTML = '';

        // Truth 필드 우선 사용
        const truth = data.truth || {};
        const bundle = data.bundle || {};
        
        // Truth 필드에서 sentence_annotations_structured 가져오기 (우선순위: truth.sentence_annotations_structured > bundle.sentence_annotations_structured > data.sentence_annotations_structured)
        const structured = Array.isArray(truth.sentence_annotations_structured || bundle.sentence_annotations_structured || data.sentence_annotations_structured)
            ? (truth.sentence_annotations_structured || bundle.sentence_annotations_structured || data.sentence_annotations_structured)
            : [];

        if (structured.length) {
            structured.forEach(item => {
                const div = document.createElement('div');
                div.className = 'sentence-item';
                const indexNum = typeof item.index === 'number' ? Math.floor(item.index) : parseInt(item.index) || 1;
                // main과 sub를 그대로 사용 - test.py에서 보낸 값 유지
                const mainLabel = item.main || '—';
                const subLabel = item.sub || '—';
                div.innerHTML = `
                    <span style="color: #808080;">${String(indexNum).padStart(2, '0')}.</span>
                    <span class="sentence-text">${item.text || ''}</span>
                    <span class="sentence-tag">[${mainLabel}|${subLabel}]</span>
                `;
                container.appendChild(div);
            });
            return;
        }

        const annotations = data.sentence_annotations || [];
        
        if (!annotations || annotations.length === 0) {
            container.innerHTML = '<div style="color: #808080;">문장 태깅 정보 없음</div>';
            return;
        }
        
        annotations.forEach(annotation => {
            const item = document.createElement('div');
            item.className = 'sentence-item';
            
            if (typeof annotation === 'string') {
                const match = annotation.match(/^(\d+)\.\s*(.+?)\s*\[(.+?)\|(.+?)\]$/);
                if (match) {
                    const [, num, text, main, sub] = match;
                    item.innerHTML = `
                        <span style="color: #808080;">${num}.</span>
                        <span class="sentence-text">${text}</span>
                        <span class="sentence-tag">[${main}|${sub}]</span>
                    `;
                } else {
                    item.textContent = annotation;
                }
            } else if (typeof annotation === 'object') {
                item.innerHTML = `
                    <span class="sentence-text">${annotation.text || ''}</span>
                    <span class="sentence-tag">[${annotation.main || '—'}|${annotation.sub || '—'}]</span>
                `;
            }
            
            container.appendChild(item);
        });
    },

    /**
     * Business View: 제품 스펙 렌더링
     * 
     * Truth 필드 연결:
     * - 제품/리포트: data.truth.products (p1/p3/p5)
     * - p1 (예측형 행동 인텔리전스): data.truth.products.p1
     * - p3 (위험 평가): data.truth.products.p3
     * - p5 (심리 프로파일 + 행동 예측): data.truth.products.p5
     */
    renderProductSpecs(data) {
        const container = this.getEl('analysisProductSpecs');
        if (!container) return;

        // Truth 필드 우선 사용
        const truth = data.truth || {};
        const bundle = data.bundle || {};
        const truthProducts = truth.products || bundle.products || {};
        
        // Truth 필드에서 products 가져오기 (우선순위: truth.products > bundle.products > data.products)
        const products = truthProducts || data.products || {};
        const domainProfile = ((truth.meta || data.meta)?.domain_profile || data?.poster?.domain_profile || '').toLowerCase();
        const isGeneric = domainProfile && domainProfile !== 'service';

        container.innerHTML = '';

        if (isGeneric) {
            const sections = Array.isArray(products.generic?.sections) ? products.generic.sections : [];
            if (!sections.length) {
                container.innerHTML = '<div class="empty-state">감정 여정에 대한 추가 인사이트가 없습니다.</div>';
                return;
            }
            sections.forEach((section, idx) => {
                const block = document.createElement('div');
                block.className = 'product-spec';
                const title = section.title || `섹션 ${idx + 1}`;
                const items = Array.isArray(section.items) ? section.items : [];
                block.innerHTML = `
                    <div class="product-title">${title}</div>
                    ${items.length ? items.map(item => `<div class="product-item">- ${item}</div>`).join('') : '<div class="product-item">- 세부 항목이 없습니다.</div>'}
                `;
                container.appendChild(block);
            });

            const highlights = Array.isArray(products.generic?.highlights) ? products.generic.highlights : [];
            if (highlights.length) {
                const highlightBlock = document.createElement('div');
                highlightBlock.className = 'product-spec';
                highlightBlock.innerHTML = `
                    <div class="product-title">하이라이트</div>
                    ${highlights.map(item => `<div class="product-item">• ${item}</div>`).join('')}
                `;
                container.appendChild(highlightBlock);
            }
            return;
        }
        
        if (products.p1 && Object.keys(products.p1).length) {
            const p1 = products.p1;
            const spec1 = document.createElement('div');
            spec1.className = 'product-spec';
            spec1.innerHTML = `
                <div class="product-title">[1] 예측형 행동 인텔리전스</div>
                <div class="product-item">- 현재 주요 감정: ${formatEmotions(p1.headline_emotions)}</div>
                <div class="product-item">- 감정 강도: ${p1.intensity ?? '—'}</div>
                <div class="product-item">- 3일 내 서비스 위험 등급: ${typeof p1.churn_probability === 'number' ? `${p1.churn_probability}%` : '—'}</div>
                <div class="product-item">- 주요 트리거: ${formatArray(p1.triggers) || '—'}</div>
                <div class="product-item">- 권장 조치: ${formatArray(p1.recommended_actions)}</div>
            `;
            container.appendChild(spec1);
        }
        
        if (products.p3 && Object.keys(products.p3).length) {
            const p3 = products.p3;
            const spec3 = document.createElement('div');
            spec3.className = 'product-spec';
            spec3.innerHTML = `
                <div class="product-title">[3] 위험 평가</div>
                <div class="product-item">- 위험 등급: ${p3.grade || 'Medium'}</div>
                <div class="product-item">- 위험 점수: ${p3.risk_score ?? '—'}</div>
                <div class="product-item">- 경고 상태: ${p3.alert ? '⚠️ 경고' : '✅ 정상'}</div>
            `;
            container.appendChild(spec3);
        }
        
        if (products.p5 && Object.keys(products.p5).length) {
            const p5 = products.p5;
            const spec5 = document.createElement('div');
            spec5.className = 'product-spec';
            spec5.innerHTML = `
                <div class="product-title">[5] 심리 프로파일 + 행동 예측</div>
                <div class="product-item">- 심리 안정성 지수: ${p5.stability ?? '—'}${typeof p5.stability === 'number' ? '%' : ''}</div>
                <div class="product-item">- 감정 성숙도: ${p5.maturity ?? '—'}${typeof p5.maturity === 'number' ? ` (${getMaturityLevel(p5.maturity)})` : ''}</div>
                <div class="product-item">- 방어기제: ${formatArray(p5.defenses) || '(감지 안 됨)'}</div>
                ${p5.scenarios && p5.scenarios.length > 0 ? `
                <div class="product-item">- 예상 행동 시나리오:</div>
                ${p5.scenarios.map(s => `
                    <div class="product-item" style="padding-left: 40px;">• ${s.name}: ${(s.prob * 100).toFixed(1)}%</div>
                `).join('')}
                ` : '<div class="product-item">- 예상 행동 시나리오: (내향/모호문: 예측 비활성화)</div>'}
            `;
            container.appendChild(spec5);
        }
    },

    renderExplainability(data) {
        const container = this.getEl('analysisExplainability');
        if (!container) return;

        const raw = data.raw_json || {};
        const results = data.results || raw.results || {};
        const aliasMap = raw.sub_label_map || {};
        const explain = data.explainability || {};
        const masterReport = data.master_report || '';

        container.classList.add('explain-grid');
        container.innerHTML = '';

        const createTitle = (title) => {
            const header = document.createElement('div');
            header.className = 'explain-card-title';
            header.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M2 12l10 5 10-5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M12 17l-5 2.5 5 2.5 5-2.5-5-2.5z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                ${title}
            `;
            return header;
        };

        const addCard = (title, bodyEl, metaText) => {
            if (!bodyEl) return;
            const card = document.createElement('div');
            card.className = 'explain-card';
            card.appendChild(createTitle(title));
            card.appendChild(bodyEl);
            if (metaText) {
                const meta = document.createElement('span');
                meta.className = 'explain-card-meta';
                meta.textContent = metaText;
                card.appendChild(meta);
            }
            container.appendChild(card);
        };

        const buildWhy = () => {
            const lines = chooseLines(explain.why_lines, getMasterSection(masterReport, '왜 이런 감정인가'));
            if (!lines.length) return null;
            const body = document.createElement('div');
            body.className = 'explain-card-body';
            // 각 라인을 구조화된 형태로 표시
            const content = document.createElement('div');
            content.style.cssText = 'display: flex; flex-direction: column; gap: 0.5rem;';
            
            // [개선] 핵심 표현 데이터 준비 (linguistic_matcher 또는 context_extractor 사용)
            let realKeyPhrases = [];
            if (results.linguistic_matcher && Array.isArray(results.linguistic_matcher.matches)) {
                realKeyPhrases = results.linguistic_matcher.matches.map(m => m.text || m.pattern).filter(Boolean);
            }
            if (!realKeyPhrases.length && results.context_extractor && Array.isArray(results.context_extractor.key_phrases)) {
                realKeyPhrases = results.context_extractor.key_phrases;
            }
            // 중복 제거 및 상위 5개
            realKeyPhrases = [...new Set(realKeyPhrases)].slice(0, 5);

            lines.forEach(line => {
                const lineDiv = document.createElement('div');
                lineDiv.style.cssText = 'line-height: 1.75;';
                
                // [Fix] 핵심 표현 라인이고, 내용이 sentence_count 등 기술 용어라면 실제 키워드로 대체
                let displayLine = line;
                if (line.includes('핵심 표현:') && (line.includes('sentence_count') || line.includes('word_count') || line.includes('emotion_intensity'))) {
                    if (realKeyPhrases.length > 0) {
                        displayLine = `핵심 표현: ${realKeyPhrases.join(', ')}`;
                    } else {
                        // 대체할 키워드가 없으면 해당 라인 숨김 (오해 방지)
                        return; 
                    }
                }

                // 강조 표시가 필요한 부분 처리
                const processedLine = displayLine
                    .replace(/(메인 감정|세부 감정|전이\(요약-SSOT\)|전이\(상세\)|핵심 표현):/g, '<strong style="color: #60a5fa;">$1:</strong>')
                    .replace(/(\+|−)/g, '<em style="color: #f59e0b;">$1</em>');
                lineDiv.innerHTML = processedLine;
                content.appendChild(lineDiv);
            });
            body.appendChild(content);
            return body;
        };

        const buildReasoning = () => {
            // ★★★ SSOT 우선: test.py → /api/analyze → data.truth.reasoning_path_lines 사용 ★★★
            const truth = data.truth || {};
            const backendLines = Array.isArray(truth.reasoning_path_lines)
                ? truth.reasoning_path_lines
                : (Array.isArray(data.reasoning_path_lines) ? data.reasoning_path_lines : []);

            const snapshotLines = Array.isArray(explain.reasoning_path_lines)
                ? explain.reasoning_path_lines
                : [];

            let lines = backendLines.length ? backendLines : snapshotLines;
            if (!lines.length) {
                lines = getMasterSection(masterReport, '추론 경로(설명가능성)') || [];
            }

            if (!lines.length) return null;
            const wrapper = document.createElement('div');
            wrapper.className = 'reasoning-step';
            lines.forEach(line => {
                const item = document.createElement('div');
                item.className = 'reasoning-item';
                item.textContent = line.trim();
                wrapper.appendChild(item);
            });
            return wrapper;
        };

        const buildTop10 = () => {
            // ★★★ SSOT 우선: test.py → /api/analyze → data.truth.sub_top10_lines 사용 ★★★
            const truth = data.truth || {};
            const backendLines = Array.isArray(truth.sub_top10_lines)
                ? truth.sub_top10_lines
                : (Array.isArray(data.sub_top10_lines) ? data.sub_top10_lines : []);

            // explainability 스냅샷에 복제본이 있을 수 있으나, Truth가 있으면 무시
            const snapshotLines = Array.isArray(explain.sub_top10_lines) ? explain.sub_top10_lines : [];

            const lines = backendLines.length ? backendLines : snapshotLines;
            if (!lines.length) return null;
            const wrapper = document.createElement('div');
            wrapper.className = 'top10-grid';
            lines.forEach(line => {
                // 형식: "분개  0.035" 또는 "                      분개  0.035" (공백으로 정렬됨)
                const trimmed = line.trim();
                // 마지막 숫자 부분을 찾기 (소수점 포함)
                const match = trimmed.match(/^(.+?)\s+([0-9]+\.[0-9]+)$/);
                if (match) {
                    const name = match[1].trim();
                    const score = match[2];
                    const row = document.createElement('div');
                    row.className = 'top10-row';
                    row.innerHTML = `<span class="top10-name">${name}</span><span class="top10-score">${score}</span>`;
                    wrapper.appendChild(row);
                } else {
                    // 파싱 실패 시 전체 라인 표시
                    const row = document.createElement('div');
                    row.className = 'top10-row';
                    row.textContent = trimmed;
                    wrapper.appendChild(row);
                }
            });
            return wrapper;
        };

        const buildKeywords = () => {
            // [개선] 핵심 표현을 한국어 키워드 위주로 표시
            // 1순위: linguistic_matcher / context_extractor에서 직접 추출
            let phrases = [];
            if (results.linguistic_matcher && Array.isArray(results.linguistic_matcher.matches)) {
                phrases = results.linguistic_matcher.matches
                    .map(m => m.text || m.pattern)
                    .filter(Boolean);
            }
            if ((!phrases || !phrases.length) && results.context_extractor && Array.isArray(results.context_extractor.key_phrases)) {
                phrases = results.context_extractor.key_phrases.filter(Boolean);
            }

            // 2.5순위: bundle.anchors.entities (남편, 급여 등 토큰)
            const truth = data.truth || {};
            const anchors = truth.anchors || {};
            if ((!phrases || !phrases.length) && Array.isArray(anchors.entities)) {
                phrases = anchors.entities
                    .map(e => e.text || e.name || e.term)
                    .filter(Boolean);
            }

            // 한글 포함 여부 체크
            const hasHangul = (s) => /[가-힣]/.test(String(s || ''));

            // 중복 제거
            if (phrases && phrases.length) {
                phrases = [...new Set(phrases)].filter(hasHangul);
            }

            let items;
            if (phrases && phrases.length) {
                // matcher/context_extractor/anchors에서 얻은 한국어 키워드
                items = phrases.map(term => ({ term }));
            } else if (Array.isArray(explain.keywords)) {
                // 다음 우선순위: explain.keywords 중 한글이 포함된 항목만 사용
                items = explain.keywords
                    .map(item => (typeof item === 'string' ? { term: item } : item))
                    .filter(item => hasHangul(item.term))
                    .slice(0, 8);
            } else {
                items = [];
            }

            if (!items.length) {
                const empty = document.createElement('div');
                empty.className = 'explain-card-body';
                empty.textContent = '핵심 표현을 찾지 못했습니다.';
                return empty;
            }

            const wrapper = document.createElement('div');
            wrapper.className = 'keyword-pills';
            items.forEach(item => {
                const term = item.term;
                if (!term) return;
                const pill = document.createElement('span');
                pill.className = 'keyword-pill';
                if (typeof item.score === 'number') {
                    pill.innerHTML = `<span>${term}</span><span>${item.score}</span>`;
                } else {
                    pill.textContent = term;
                }
                wrapper.appendChild(pill);
            });
            return wrapper;
        };

        const buildMatched = () => {
            // ★★★ test.py 원본 데이터 우선 사용 ★★★
            // 1순위: results.emotion_classification.matched_phrases
            // 2순위: sentence_annotations_structured (문장별 감정 태깅에서 매칭)
            const truth = data.truth || {};
            const matched = Array.isArray(results.emotion_classification?.matched_phrases)
                ? results.emotion_classification.matched_phrases
                : [];

            // 문장별 감정 태깅 데이터 (실제 문장 텍스트를 가져오기 위해)
            const sentenceAnnotations = Array.isArray(truth.sentence_annotations_structured || data.sentence_annotations_structured)
                ? (truth.sentence_annotations_structured || data.sentence_annotations_structured)
                : [];

            // sentence_annotations_structured를 우선 사용하고,
            // 없을 때만 matched_phrases로 폴백
            let finalMatched = [];
            if (sentenceAnnotations.length > 0) {
                // 중복 제거: 같은 문장 텍스트는 한 번만 표시
                const seenFromSentences = new Set();
                finalMatched = sentenceAnnotations
                    .filter(s => {
                        const text = s.text || '';
                        if (!text || seenFromSentences.has(text)) {
                            return false;
                        }
                        seenFromSentences.add(text);
                        return true;
                    })
                    .slice(0, 5)
                    .map(s => ({
                        main_emotion: s.main || '',
                        emotion_category: s.main || '',
                        sub_emotion: s.sub || '',
                        sub: s.sub || '',
                        text: s.text || '',
                        evidence_sentence: s.text || '',
                        pattern: s.text ? s.text.substring(0, 50) : '',
                        explanation: s.text || '',
                        confidence: typeof s.confidence === 'number' ? s.confidence : 0.7,
                        confidence_pct: typeof s.confidence === 'number' ? (s.confidence * 100) : 70
                    }));
            } else if (matched.length > 0) {
                // matched_phrases가 있을 때만 폴백 사용 (문장 텍스트는 sentence_annotations로 보완)
                finalMatched = matched.slice(0, 5).map(item => {
                    const matchedItem = { ...item };

                    if (!matchedItem.evidence_sentence && !matchedItem.text && sentenceAnnotations.length > 0) {
                        const subEmotion = matchedItem.sub_emotion || matchedItem.sub || '';
                        const pattern = matchedItem.pattern || matchedItem.explanation || '';

                        const matchedSentence = sentenceAnnotations.find(s => {
                            const sSub = s.sub || '';
                            return sSub && subEmotion && (
                                sSub.includes(subEmotion) ||
                                subEmotion.includes(sSub) ||
                                sSub === subEmotion
                            );
                        });

                        if (matchedSentence && matchedSentence.text) {
                            matchedItem.evidence_sentence = matchedSentence.text;
                            matchedItem.text = matchedSentence.text;
                        } else if (sentenceAnnotations.length > 0) {
                            matchedItem.evidence_sentence = sentenceAnnotations[0].text || '';
                            matchedItem.text = sentenceAnnotations[0].text || '';
                        }
                    }

                    return matchedItem;
                });
            }

            // 중복 제거: 같은 문장 텍스트는 한 번만 표시
            const seenTexts = new Set();
            finalMatched = finalMatched.filter(item => {
                const text = item.evidence_sentence || item.text || '';
                if (!text || seenTexts.has(text)) {
                    return false;
                }
                seenTexts.add(text);
                return true;
            });

            if (!finalMatched.length) return null;

            // main_distribution에서 가장 높은 메인 감정 추론 (truth.main_dist 우선)
            const mainDist = truth.main_dist || data.poster?.main_distribution || data.main_distribution || {};
            const topMain = Object.keys(mainDist).length > 0
                ? Object.keys(mainDist).sort((a, b) => (mainDist[b] || 0) - (mainDist[a] || 0))[0]
                : null;
            
            const list = document.createElement('div');
            list.className = 'reasoning-step';
            finalMatched.forEach(item => {
                // main_emotion이 없거나 잘못된 경우 main_distribution 기반으로 추론
                let main = mapMainLabel(item.main_emotion || item.emotion_category);
                if (!main || main === '—' || (topMain && mainDist[main] < (mainDist[topMain] || 0) * 0.3)) {
                    // main_distribution에서 가장 높은 메인 감정 사용
                    main = topMain || main || '노';
                }
                const sub = mapSubLabel(main, item.sub_emotion || item.sub, aliasMap);
                const confidence = typeof item.confidence_pct === 'number'
                    ? `${item.confidence_pct.toFixed(1)}%`
                    : (typeof item.confidence === 'number' ? `${Math.round(item.confidence * 100)}%` : '');
                const distribution = typeof item.score_pct === 'number'
                    ? `${item.score_pct.toFixed(1)}%`
                    : '';
                
                // 실제 문장 텍스트 우선순위: evidence_sentence > text > pattern > explanation
                const sentenceText = item.evidence_sentence || item.text || item.pattern || item.explanation || '';
                
                // description은 패턴이나 설명 (문장이 아닌)
                const description = item.explanation || item.pattern || '';
                
                // 문장 텍스트가 있으면 표시
                const displayText = sentenceText || description;

                const row = document.createElement('div');
                row.className = 'reasoning-item';
                
                // 메인 라벨
                const mainLabel = document.createElement('div');
                mainLabel.style.cssText = 'display: flex; align-items: center; gap: 0.5rem; font-weight: 600; color: #93c5fd;';
                mainLabel.innerHTML = `<span>${sub ? `${main} | ${sub}` : main}</span>`;
                
                // 메타 정보
                if (distribution || confidence) {
                    const metaDiv = document.createElement('div');
                    metaDiv.className = 'explain-card-meta';
                    metaDiv.style.cssText = 'margin-top: 0.25rem;';
                    const metaParts = [];
                    if (distribution) metaParts.push(`분포 비중 ${distribution}`);
                    if (confidence) metaParts.push(`근거 신뢰도 ${confidence}`);
                    metaDiv.textContent = metaParts.join(' · ');
                    mainLabel.appendChild(metaDiv);
                }
                
                row.appendChild(mainLabel);
                
                // 문장 텍스트
                if (displayText) {
                    const textDiv = document.createElement('div');
                    textDiv.style.cssText = 'color: #e2e8f0; margin-top: 0.5rem; line-height: 1.6; font-size: 0.8125rem;';
                    textDiv.textContent = truncateText(displayText, 120);
                    row.appendChild(textDiv);
                }
                
                // 패턴 설명
                if (description && description !== displayText) {
                    const patternDiv = document.createElement('div');
                    patternDiv.className = 'explain-card-meta';
                    patternDiv.style.cssText = 'margin-top: 0.375rem; font-style: italic;';
                    patternDiv.textContent = `패턴: ${truncateText(description, 90)}`;
                    row.appendChild(patternDiv);
                }
                
                list.appendChild(row);
            });
            return list;
        };

        addCard('왜 이런 감정인가', buildWhy());
        addCard('추론 경로(설명가능성)', buildReasoning());
        addCard('핵심 문장 패턴', buildMatched(), '패턴과 문장 근거를 기반으로 감정을 추론했습니다.');
        addCard('핵심 표현', buildKeywords());

        if (!container.childElementCount) {
            container.innerHTML = '<div class="empty-state">설명 가능성 정보가 없습니다.</div>';
        }
    },

    renderModelNarrative(data) {
        const container = this.getEl('analysisModelNarrative');
        if (!container) return;

        const modeLabel = (data.meta && typeof data.meta.mode === 'string')
            ? data.meta.mode.toUpperCase()
            : (typeof data.mode === 'string' ? data.mode.toUpperCase() : 'BALANCED');

        if (modeLabel !== 'BALANCED') {
            container.innerHTML = '<div class="empty-state">BALANCED 모드에서 제공됩니다.</div>';
            return;
        }

        const masterReport = data && typeof data.master_report === 'string' ? data.master_report : '';
        const fallback = getMasterSection(masterReport, '모델A 추론');
        const primary = Array.isArray(data?.model_narrative)
            ? data.model_narrative.filter(line => typeof line === 'string' && line.trim())
            : [];
        const lines = fallback.length ? fallback : primary;

        if (!lines.length) {
            container.innerHTML = '<div class="empty-state">모델A 추론 정보가 없습니다.</div>';
            return;
        }

        const pre = document.createElement('pre');
        pre.className = 'explain-pre';
        pre.textContent = lines.join('\n');

        container.innerHTML = '';
        container.appendChild(pre);
    },

    buildSmartInsightSummary(data) {
        const lines = [];
        const truth = data.truth || {};
        const bundle = data.bundle || {};
        const explain = data.explainability || {};
        const existing = Array.isArray(data.insight_summary)
            ? data.insight_summary.filter(line => typeof line === 'string' && line.trim())
            : [];

        const seen = new Set();
        const addLine = (line) => {
            if (typeof line !== 'string') return;
            const t = line.trim();
            if (!t) return;
            if (seen.has(t)) return;
            seen.add(t);
            lines.push(t);
        };

        const mainDist = truth.main_dist
            || bundle.main_dist
            || (data.poster && data.poster.main_distribution)
            || data.main_distribution
            || {};
        const mainKeys = Object.keys(mainDist || {}).filter(k => typeof mainDist[k] === 'number');
        if (mainKeys.length) {
            const sorted = [...mainKeys].sort((a, b) => (mainDist[b] || 0) - (mainDist[a] || 0));
            const main = sorted[0];
            const rawMainVal = mainDist[main] || 0;
            const mainPct = rawMainVal <= 1 ? rawMainVal * 100 : rawMainVal;

            let topSubLine = '';
            const truthSubTop = Array.isArray(truth.sub_top) ? truth.sub_top : null;
            const explainSubTop = Array.isArray(explain.sub_top) ? explain.sub_top : null;
            const subTop = truthSubTop && truthSubTop.length ? truthSubTop : (explainSubTop || []);
            if (subTop && subTop.length) {
                const topSub = subTop[0] || {};
                const subLabel = topSub.label || topSub.sub || topSub.name || '';
                let subPct = null;
                if (typeof topSub.pct === 'number') {
                    subPct = topSub.pct;
                } else if (typeof topSub.p === 'number') {
                    subPct = topSub.p;
                } else if (typeof topSub.score === 'number') {
                    subPct = topSub.score <= 1 ? topSub.score * 100 : topSub.score;
                }
                if (subLabel) {
                    if (typeof subPct === 'number') {
                        topSubLine = `${subLabel} ${subPct.toFixed(1)}%`;
                    } else {
                        topSubLine = subLabel;
                    }
                }
            }

            let mainLine = '';
            if (topSubLine) {
                mainLine = `주 감정은 ${main}(${mainPct.toFixed(1)}%)이며, 상위 세부 감정은 ${topSubLine}입니다.`;
            } else {
                mainLine = `주 감정은 ${main}(${mainPct.toFixed(1)}%)로 파악됩니다.`;
            }
            addLine(mainLine);
        }

        const products = data.products || bundle.products || {};
        const p1 = products.p1 || {};
        const intensity = p1.intensity || '';
        const churn = typeof p1.churn_probability === 'number' ? p1.churn_probability : null;
        const horizon = typeof p1.horizon_days === 'number' ? p1.horizon_days : null;
        const triggersSrc = Array.isArray(p1.triggers) ? p1.triggers : [];
        const triggers = triggersSrc.filter(t => typeof t === 'string' && t.trim());

        const riskParts = [];
        if (intensity) {
            riskParts.push(`감정 강도는 '${intensity}'`);
        }
        if (typeof churn === 'number') {
            riskParts.push(`서비스 이탈 위험은 ${Math.round(churn)}% 수준`);
        }
        if (typeof horizon === 'number' && horizon > 0) {
            riskParts.push(`${horizon}일 이내 단기 변동 기준`);
        }

        let riskLine = '';
        if (riskParts.length) {
            riskLine = riskParts.join(', ') + '입니다.';
        }
        if (triggers.length) {
            const topTriggers = triggers.slice(0, 3).join(', ');
            const triggerText = `주요 트리거는 ${topTriggers} 입니다.`;
            riskLine = riskLine ? `${riskLine} ${triggerText}` : triggerText;
        }
        if (riskLine) {
            addLine(riskLine);
        }

        const flowSSOT = truth.flow_ssot || bundle.flow_ssot;
        if (typeof flowSSOT === 'string') {
            const trimmed = flowSSOT.trim();
            if (trimmed && trimmed !== '흐름 정보 없음') {
                const compact = trimmed.replace(/\s+/g, ' ');
                addLine(`감정 흐름은 ${compact} 패턴이 관찰됩니다.`);
        }
        }

        const truthWhy = Array.isArray(truth.why_lines) ? truth.why_lines : [];
        const truthReasoning = Array.isArray(truth.reasoning_path_lines) ? truth.reasoning_path_lines : [];
        const explainWhy = Array.isArray(explain.why_lines) ? explain.why_lines : [];
        const explainReasoning = Array.isArray(explain.reasoning_path_lines) ? explain.reasoning_path_lines : [];
        const whyLines = truthWhy.length ? truthWhy : explainWhy;
        const reasoningLines = truthReasoning.length ? truthReasoning : explainReasoning;

        const modelLines = Array.isArray(data.model_narrative)
            ? data.model_narrative.filter(line => typeof line === 'string' && line.trim())
            : [];

        const candidateGroups = [
            existing,
            modelLines,
            whyLines,
            reasoningLines,
        ];

        for (const group of candidateGroups) {
            if (!Array.isArray(group)) continue;
            for (const line of group) {
                if (lines.length >= 5) break;
                addLine(line);
            }
            if (lines.length >= 5) break;
        }

        return lines.slice(0, 5);
    },

    renderInsightSummary(data) {
        const container = this.getEl('analysisInsightSummary');
        if (!container) return;

        container.innerHTML = '';
        const summary = this.buildSmartInsightSummary(data);

        if (!Array.isArray(summary) || summary.length === 0) {
            container.innerHTML = '<div class="empty-state">요약 정보를 찾을 수 없습니다.</div>';
            return;
        }

        summary.forEach(line => {
            const item = document.createElement('div');
            item.className = 'insight-item';
            item.textContent = line;
            container.appendChild(item);
        });
    },

    renderInvestorHighlights(data) {
        const container = this.getEl('analysisInvestorHighlights');
        if (!container) return;

        const highlights = Array.isArray(data.investor_highlights) ? data.investor_highlights : [];
        if (!highlights.length) {
            container.innerHTML = '<div class="empty-state">투자 관점의 하이라이트 정보가 없습니다.</div>';
            return;
        }

        const list = document.createElement('ul');
        list.className = 'investor-list';
        highlights.forEach(text => {
            const li = document.createElement('li');
            li.textContent = text;
            list.appendChild(li);
        });

        container.innerHTML = '';
        container.appendChild(list);
    },

    renderStrategicBrief(data) {
        const container = this.getEl('analysisStrategicBrief');
        if (!container) return;

        container.innerHTML = '';

        const moduleResultsRaw = data.module_results || data.results || {};
        const moduleResults = (moduleResultsRaw && typeof moduleResultsRaw === 'object') ? moduleResultsRaw : {};
        const moduleDetails = Array.isArray(data.module_details) ? data.module_details : [];
        const moduleDetailMap = new Map(moduleDetails.map(detail => [detail.name, detail]));
        const moduleHitRate = data.module_hit_rate || {};

        const orderedNames = moduleDetails.length
            ? moduleDetails.map(detail => detail.name)
            : Array.from(new Set([
                ...Object.keys(MODULE_DISPLAY_INFO),
                ...Object.keys(moduleResults),
                ...Object.keys(moduleHitRate),
            ])).filter(Boolean);

        if (!orderedNames.length) {
            container.innerHTML = '<div style="color: #808080; text-align: center; padding: 20px;">모듈 분석 결과를 불러올 수 없습니다.</div>';
            return;
        }

        const statusCounts = orderedNames.reduce((acc, name) => {
            const detail = moduleDetailMap.get(name);
            const status = detail?.status || (moduleHitRate[name] ? 'ok' : 'missing');
            if (status === 'ok') acc.ok += 1;
            return acc;
        }, { ok: 0 });

        const summary = document.createElement('div');
        summary.style.cssText = 'margin-bottom: 12px; color: #a5b4fc; font-size: 13px;';
        summary.textContent = `모듈 커버리지 ${statusCounts.ok}/${orderedNames.length}개 가동`;
        container.appendChild(summary);

        const grid = document.createElement('div');
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = '1fr 1fr';
        grid.style.gap = '16px';

        const formatSample = (item) => {
            if (typeof item === 'string') return truncateText(item, 60);
            if (item && typeof item === 'object') {
                const text = item.text || item.pattern || item.phrase || item.excerpt || item.summary;
                const main = item.main_emotion || item.main || item.emotion || '';
                const sub = item.sub_emotion || item.sub || '';
                if (text) {
                    const label = main ? `[${main}${sub ? `|${sub}` : ''}]` : '';
                    return `${truncateText(text, 60)} ${label}`.trim();
                }
                return truncateText(JSON.stringify(item), 60);
            }
            return String(item);
        };

        const formatComplexItems = (value) => {
            if (!value) return [];
            if (Array.isArray(value)) {
                return value.slice(0, 3).map(item => {
                    if (typeof item === 'string') return item;
                    if (item && typeof item === 'object') {
                        const main = item.emotion_id || item.main || item.category || item.emotion || '';
                        let topSub = '';
                        if (Array.isArray(item.sub_emotions) && item.sub_emotions.length) {
                            const sortedSubs = [...item.sub_emotions].sort((a, b) => {
                                const sa = typeof a.score === 'number' ? a.score : (typeof a.share === 'number' ? a.share : 0);
                                const sb = typeof b.score === 'number' ? b.score : (typeof b.share === 'number' ? b.share : 0);
                                return sb - sa;
                            });
                            const first = sortedSubs[0] || {};
                            topSub = first.name || first.label || '';
                        }
                        const fallbackLabel = item.label || item.name || item.id || '';
                        let label = '';
                        if (main && topSub) {
                            label = `${main}-${topSub}`;
                        } else if (topSub) {
                            label = topSub;
                        } else if (main) {
                            label = main;
                        } else {
                            label = fallbackLabel;
                        }

                        let rawScore = null;
                        if (typeof item.score === 'number') {
                            rawScore = item.score;
                        } else if (typeof item.value === 'number') {
                            rawScore = item.value;
                        } else if (typeof item.confidence === 'number') {
                            rawScore = item.confidence;
                        }
                        const score = rawScore !== null
                            ? (rawScore <= 1 ? `${(rawScore * 100).toFixed(1)}%` : rawScore.toFixed(1))
                            : '';
                        const safeLabel = label || '기타';
                        return `${safeLabel}${score ? ` ${score}` : ''}`.trim();
                    }
                    return String(item);
                });
            }
            if (typeof value === 'object') {
                return Object.entries(value).slice(0, 3).map(([k, v]) => {
                    if (typeof v === 'number') {
                        const pct = (v <= 1 ? v * 100 : v).toFixed(1);
                        return `${k} ${pct}%`;
                    }
                    if (v && typeof v === 'object') {
                        const score = typeof v.score === 'number'
                            ? (v.score <= 1 ? `${(v.score * 100).toFixed(1)}%` : v.score.toFixed(1))
                            : '';
                        const label = k || v.label || v.name || '기타';
                        return `${label}${score ? ` ${score}` : ''}`.trim();
                    }
                    return `${k}: ${String(v)}`;
                });
            }
            return [];
        };

        const buildModuleInsight = (key, moduleData) => {
            if (!moduleData || typeof moduleData !== 'object') return '';
            switch (key) {
                case 'linguistic_matcher': {
                    const matches = moduleData.matches || moduleData.matched_phrases || [];
                    if (Array.isArray(matches) && matches.length) {
                        const samples = matches.slice(0, 3).map(formatSample);
                        return `<strong>매칭 ${matches.length}개</strong><br/>${samples.map(s => `• ${s}`).join('<br/>')}`;
                    }
                    return '키워드 매칭 결과 없음';
                }
                case 'pattern_extractor': {
                    const parts = [];
                    const emotionLabels = {'희': '긍정', '노': '분노', '애': '슬픔', '락': '즐거움'};
                    
                    // ★★★ 실제 데이터 구조: emotion_progression[].emotions[] ★★★
                    const temporalAnalysis = moduleData.temporal_analysis || {};
                    const emotionProgression = temporalAnalysis.emotion_progression || [];
                    
                    if (Array.isArray(emotionProgression) && emotionProgression.length > 0) {
                        // 모든 문장의 감정을 집계
                        const emotionScores = {};
                        const subEmotionList = [];
                        
                        for (const sentence of emotionProgression) {
                            if (!sentence || !Array.isArray(sentence.emotions)) continue;
                            
                            for (const emo of sentence.emotions) {
                                const primary = emo.primary || '';
                                const subEmo = emo.sub_emotion || '';
                                const score = emo.score || 0;
                                const conf = emo.confidence || 0;
                                
                                if (primary) {
                                    if (!emotionScores[primary]) {
                                        emotionScores[primary] = { total: 0, count: 0, maxConf: 0 };
                                    }
                                    emotionScores[primary].total += score;
                                    emotionScores[primary].count += 1;
                                    emotionScores[primary].maxConf = Math.max(emotionScores[primary].maxConf, conf);
                                    
                                    // 세부 감정 기록
                                    if (subEmo && score > 0) {
                                        subEmotionList.push({ primary, subEmo, score, conf });
                                    }
                                }
                            }
                        }
                        
                        // 점수 기준 정렬
                        const sorted = Object.entries(emotionScores)
                            .map(([emo, data]) => ({
                                emo,
                                label: emotionLabels[emo] || emo,
                                totalScore: data.total,
                                count: data.count,
                                maxConf: data.maxConf
                            }))
                            .filter(e => e.totalScore > 0)
                            .sort((a, b) => b.totalScore - a.totalScore);
                        
                        if (sorted.length > 0) {
                            parts.push(`<strong>감정 패턴 ${sorted.length}개 감지</strong>`);
                            
                            // 상위 감정 + 세부 감정 표시
                            const topSubEmotions = subEmotionList
                                .sort((a, b) => b.score - a.score)
                                .slice(0, 3);
                            
                            const lines = topSubEmotions.map(e => {
                                const label = emotionLabels[e.primary] || e.primary;
                                return `• ${label}-${e.subEmo}: ${e.score.toFixed(1)}점 (${Math.round(e.conf * 100)}%)`;
                            });
                            parts.push(lines.join('<br/>'));
                        }
                        
                        // 문장별 감정 흐름 (지배 감정)
                        const flow = emotionProgression
                            .filter(s => s && Array.isArray(s.emotions) && s.emotions.length > 0)
                            .slice(0, 4)
                            .map(s => {
                                const topEmo = s.emotions.reduce((a, b) => (b.score > a.score ? b : a), s.emotions[0]);
                                return emotionLabels[topEmo.primary] || topEmo.primary;
                            });
                        
                        if (flow.length > 1) {
                            parts.push(`흐름: ${flow.join(' → ')}`);
                        }
                    }
                    
                    // 신뢰도 표시
                    if (typeof moduleData.confidence === 'number' && moduleData.confidence > 0) {
                        parts.push(`신뢰도: ${Math.round(moduleData.confidence * 100)}%`);
                    }
                    
                    if (parts.length > 0) {
                        return parts.join('<br/>');
                    }
                    
                    return moduleData.summary || '';
                }
                case 'context_analysis': {
                    // ★★★ 트리거 키 한글 번역 매핑 ★★★
                    const triggerKeyKorean = {
                        'negative': '부정 감정',
                        'adversatives': '역접/전환',
                        'recovery': '회복 신호',
                        'cancel': '취소/환불',
                        'positive': '긍정 감정',
                        'conflict': '갈등 표현'
                    };
                    
                    let triggers = moduleData.top_triggers || moduleData.triggers || [];
                    if (!Array.isArray(triggers)) {
                        triggers = [];
                    }
                    if (!triggers.length) {
                        const globalTriggers = Array.isArray(data.triggers) ? data.triggers : [];
                        if (globalTriggers.length) {
                            triggers = globalTriggers;
                        } else {
                            const truthTriggers = data.truth && data.truth.triggers;
                            const bundleTriggers = data.bundle && data.bundle.triggers;
                            const source = (truthTriggers && typeof truthTriggers === 'object')
                                ? truthTriggers
                                : (bundleTriggers && typeof bundleTriggers === 'object' ? bundleTriggers : null);
                            if (source) {
                                // ★★★ 수정: 키 대신 실제 값을 추출하고, 값이 없으면 한글 키 이름 사용 ★★★
                                const extractedTriggers = [];
                                for (const key of Object.keys(source)) {
                                    const items = source[key];
                                    if (Array.isArray(items) && items.length > 0) {
                                        // 해당 카테고리에 실제 트리거가 있으면 추출
                                        items.forEach(item => {
                                            const text = typeof item === 'string' ? item : (item && item.text ? item.text : null);
                                            if (text && text.trim()) {
                                                extractedTriggers.push({ text: text.trim(), category: triggerKeyKorean[key] || key });
                                            }
                                        });
                                    }
                                }
                                // 실제 트리거가 추출되었으면 사용, 없으면 카테고리 이름(한글) 사용
                                if (extractedTriggers.length > 0) {
                                    triggers = extractedTriggers.slice(0, 5);
                                } else {
                                    // 실제 트리거가 없으면 카테고리 키를 한글로 번역하여 표시
                                    const keys = Object.keys(source).filter(k => {
                                        const v = source[k];
                                        return v !== null && v !== undefined && (Array.isArray(v) ? v.length > 0 : true);
                                    }).slice(0, 5);
                                    triggers = keys.map(t => ({ text: triggerKeyKorean[t] || t }));
                                }
                            }
                        }
                    }
                    if (Array.isArray(triggers) && triggers.length) {
                        const items = triggers.slice(0, 3).map(t => {
                            if (typeof t === 'string') return t;
                            if (t && typeof t === 'object') {
                                const text = t.text || t.term || t.keyword || '';
                                // 카테고리 정보가 있으면 함께 표시
                                if (t.category && text !== t.category) {
                                    return `${text} (${t.category})`;
                                }
                                return text;
                            }
                            return String(t);
                        }).filter(s => s && s.trim());
                        if (items.length) {
                            return `<strong>트리거 ${items.length}개</strong><br/>${items.map(s => `• ${s}`).join('<br/>')}`;
                        }
                    }
                    if (moduleData.summary) return moduleData.summary;
                    return '';
                }
                case 'context_extractor': {
                    const anchors = (data.bundle && data.bundle.anchors) || {};
                    const entitiesSrc = Array.isArray(moduleData.entities)
                        ? moduleData.entities
                        : (Array.isArray(anchors.entities) ? anchors.entities : []);
                    if (entitiesSrc.length) {
                        const entities = entitiesSrc.slice(0, 3).map(e =>
                            typeof e === 'object' ? (e.text || e.name || e.value || JSON.stringify(e)) : String(e)
                        );
                        return `<strong>엔티티 ${entitiesSrc.length}개</strong><br/>${entities.map(e => `• ${truncateText(e, 50)}`).join('<br/>')}`;
                    }

                    const keyPhrases = Array.isArray(moduleData.key_phrases)
                        ? moduleData.key_phrases
                        : (Array.isArray(anchors.key_phrases) ? anchors.key_phrases : []);

                    const parts = [];
                    let sentenceCount = typeof moduleData.sentence_count === 'number' ? moduleData.sentence_count : null;
                    if (sentenceCount == null) {
                        const truthSentences = (data.truth && Array.isArray(data.truth.sentence_annotations_structured))
                            ? data.truth.sentence_annotations_structured.length
                            : (Array.isArray(data.sentence_annotations_structured) ? data.sentence_annotations_structured.length : null);
                        if (typeof truthSentences === 'number' && truthSentences > 0) {
                            sentenceCount = truthSentences;
                        }
                    }
                    if (typeof sentenceCount === 'number' && sentenceCount > 0) {
                        parts.push(`${sentenceCount}개 문장 기반 맥락 추출`);

                        // 어떤 문장들이 맥락 분석에 사용되었는지 간단한 예시를 함께 표시
                        const sentRows = (data.truth && Array.isArray(data.truth.sentence_annotations_structured))
                            ? data.truth.sentence_annotations_structured
                            : (Array.isArray(data.sentence_annotations_structured) ? data.sentence_annotations_structured : []);
                        if (Array.isArray(sentRows) && sentRows.length) {
                            const examples = sentRows
                                .slice(0, Math.min(3, sentRows.length))
                                .map((row, idx) => {
                                    const t = (row && typeof row.text === 'string') ? row.text : '';
                                    if (!t) return null;
                                    return `문장 ${idx + 1}: ${truncateText(t, 70)}`;
                                })
                                .filter(Boolean);
                            if (examples.length) {
                                parts.push(examples.join('<br/>'));
                            }
                        }
                    }
                    const flow = moduleData.dominant_flow;
                    if (Array.isArray(flow) && flow.length) {
                        const flowText = flow.slice(0, 3).map(item => {
                            if (Array.isArray(item) && item.length >= 2) {
                                const emo = item[0];
                                const val = typeof item[1] === 'number' ? item[1] : 0;
                                const pct = val <= 1 ? (val * 100).toFixed(1) : val.toFixed(1);
                                return `${emo} ${pct}%`;
                            }
                            if (item && typeof item === 'object') {
                                const emo = item.emo || item.label || '';
                                const raw = typeof item.score === 'number' ? item.score : (typeof item.value === 'number' ? item.value : 0);
                                const pct = raw <= 1 ? (raw * 100).toFixed(1) : raw.toFixed(1);
                                return `${emo} ${pct}%`;
                            }
                            return String(item);
                        }).join(', ');
                        parts.push(`감정 흐름: ${flowText}`);
                    }
                    if (keyPhrases.length) {
                        const phrases = keyPhrases.slice(0, 3).map(p =>
                            typeof p === 'string' ? p : (p.text || p.term || JSON.stringify(p))
                        );
                        parts.push(`핵심 키워드: ${phrases.join(', ')}`);
                    }
                    if (parts.length) {
                        return parts.join('<br/>');
                    }
                    if (moduleData.context) {
                        return `컨텍스트: ${truncateText(JSON.stringify(moduleData.context), 70)}`;
                    }
                    return '';
                }
                case 'time_series_analyzer': {
                    // ★★★ 수정: 파이프라인 실제 반환 키 사용 ★★★
                    const emotionSeq = moduleData.emotion_sequence || moduleData.sequence_analysis || [];
                    const seriesData = moduleData.series || [];
                    const timeFlow = moduleData.time_flow || {};
                    const causeEffect = moduleData.cause_effect || [];
                    
                    // 시퀀스 카운트
                    let seqCount = 0;
                    if (Array.isArray(emotionSeq) && emotionSeq.length) {
                        seqCount = emotionSeq.length;
                    } else if (Array.isArray(seriesData) && seriesData.length) {
                        seqCount = seriesData.length;
                    }
                    
                    const flowMode = timeFlow.mode || '';
                    const hasCausality = Array.isArray(causeEffect) && causeEffect.length > 0;
                    
                    if (seqCount > 1 || flowMode === 'linear' || flowMode === 'linear_capped') {
                        const parts = [`<strong>시계열 ${seqCount}구간 분석</strong>`];
                        
                        // 시간 흐름 모드 표시
                        if (flowMode) {
                            const modeDesc = {
                                'linear': '정상 흐름',
                                'linear_capped': '보수적 캡',
                                'static': '정지'
                            }[flowMode] || flowMode;
                            parts.push(`흐름: ${modeDesc}`);
                        }
                        
                        // 감정 변화 표시: emotion_sequence에서 직접 추출
                        // 복잡한 키에서 주요 감정 집계 헬퍼
                        const getDominantEmotion = (emotionsDict) => {
                            if (!emotionsDict || !Object.keys(emotionsDict).length) return null;
                            const mainCounts = {'희': 0, '노': 0, '애': 0, '락': 0};
                            for (const [key, val] of Object.entries(emotionsDict)) {
                                // '노-분개-sentiment_analysis' → '노'
                                const firstPart = key.includes('-') ? key.split('-')[0] : key;
                                if (firstPart in mainCounts) {
                                    mainCounts[firstPart] += val;
                                }
                            }
                            const maxVal = Math.max(...Object.values(mainCounts));
                            if (maxVal > 0) {
                                return Object.entries(mainCounts).find(([k, v]) => v === maxVal)?.[0];
                            }
                            return null;
                        };
                        
                        // emotion_sequence에서 감정 변화 추출
                        const emotionChanges = [];
                        if (Array.isArray(emotionSeq) && emotionSeq.length >= 2) {
                            let prevEmo = null;
                            emotionSeq.forEach(es => {
                                const emotions = es?.emotions || {};
                                const currEmo = getDominantEmotion(emotions);
                                if (currEmo && prevEmo && currEmo !== prevEmo) {
                                    emotionChanges.push([prevEmo, currEmo]);
                                }
                                prevEmo = currEmo;
                            });
                        }
                        
                        // 중복 제거 및 표시
                        const uniqueChanges = [...new Set(emotionChanges.map(c => c.join('→')))].map(s => s.split('→'));
                        if (uniqueChanges.length > 0) {
                            parts.push(`감정 변화 ${uniqueChanges.length}건`);
                            uniqueChanges.slice(0, 2).forEach(([from, to]) => {
                                parts.push(`• ${from} → ${to}`);
                            });
                        }
                        
                        return parts.join('<br/>');
                    } else if (seqCount === 1) {
                        return '<strong>순간적 감정 스냅샷</strong><br/>현재 시점 감정에 집중';
                    } else if (timeFlow || Object.keys(moduleData.summary || {}).length) {
                        return '<strong>시계열 패턴 분석됨</strong><br/>감정 변화 패턴 감지';
                    }
                    return '';
                }
                case 'transition_analyzer': {
                    let transitions = [];
                    if (Array.isArray(data.transitions_structured) && data.transitions_structured.length) {
                        transitions = data.transitions_structured;
                    } else if (Array.isArray(moduleData.transitions) && moduleData.transitions.length) {
                        transitions = moduleData.transitions;
                    } else {
                        const rawResults = (data.raw_json || {}).results || {};
                        const rawModule = rawResults.transition_analyzer || {};
                        if (Array.isArray(rawModule.transitions) && rawModule.transitions.length) {
                            transitions = rawModule.transitions;
                        }
                    }
                    if (transitions.length) {
                        const items = transitions.slice(0, 3).map((t) => {
                            if (typeof t === 'string') return t;
                            if (t && typeof t === 'object') {
                                const fromMain = t.from_main || t.from || t.from_emotion || '?';
                                // [GENIUS FIX] Late Binding Normalization
                                // 렌더링 직전에 한 번 더 검사하여 sub_ 제거
                                let fromSubRaw = t.from_sub || '';
                                if (fromSubRaw === '—' || fromSubRaw.includes('sub_')) {
                                     fromSubRaw = normalizeSubLabel(fromMain, fromSubRaw);
                                }
                                const fromSub = fromSubRaw && fromSubRaw !== '—' ? `(${fromSubRaw})` : '';
                                
                                const toMain = t.to_main || t.to || t.to_emotion || '?';
                                let toSubRaw = t.to_sub || '';
                                if (toSubRaw === '—' || toSubRaw.includes('sub_')) {
                                     toSubRaw = normalizeSubLabel(toMain, toSubRaw);
                                }
                                const toSub = toSubRaw && toSubRaw !== '—' ? `(${toSubRaw})` : '';
                                
                                const trigger = t.trigger || t.transition_reason;
                                return `${fromMain}${fromSub} → ${toMain}${toSub}${trigger ? ` · ${truncateText(trigger, 60)}` : ''}`;
                            }
                            return String(t);
                        });
                        return `<strong>전이 ${transitions.length}개</strong><br/>${items.map(s => `• ${s}`).join('<br/>')}`;
                    }
                    return '';
                }
                case 'relationship_analyzer': {
                    // [GENIUS FIX] Relationship Data Synchronization
                    // 이 모듈은 문장 간 관계를 보여주지만, 감정 라벨이 Truth Data(test.py 결과)와 
                    // 불일치하는 문제가 있었습니다 (예: 5번 모듈은 '공격성', 8번 모듈은 '감사'로 태깅).
                    // 따라서 문장 텍스트를 기준으로 Truth Data의 감정 라벨을 강제 적용(Override)하여 일관성을 보장합니다.
                    
                    const sentencesRaw = Array.isArray(moduleData.sentences) ? moduleData.sentences : [];
                    const anchors = (data.bundle && data.bundle.anchors) || {};
                    const sentencesFallback = Array.isArray(anchors.sentences) ? anchors.sentences : [];
                    const sentences = sentencesRaw.length ? sentencesRaw : sentencesFallback;

                    // 1. Truth Data 매핑 테이블 생성 (문장 텍스트 -> 정확한 감정)
                    const truthMap = new Map();
                    const truthAnnotations = data.truth?.sentence_annotations_structured 
                        || data.sentence_annotations_structured 
                        || [];
                    
                    if (Array.isArray(truthAnnotations)) {
                        truthAnnotations.forEach(item => {
                            if (item && item.text) {
                                // 공백 제거 후 매핑하여 매칭 확률 높임
                                const key = item.text.trim().replace(/\s+/g, ' ');
                                truthMap.set(key, {
                                    main: mapMainLabel(item.main || item.main_emotion),
                                    sub: normalizeSubLabel(item.main, item.sub_label || item.sub)
                                });
                            }
                        });
                    }

                    if (sentences.length) {
                        const samples = sentences.slice(0, 3).map((s) => {
                            const text = typeof s === 'string' ? s : (s.text || s.sentence || s.raw || '');
                            const key = text.trim().replace(/\s+/g, ' ');
                            
                            // 2. Truth Data 우선 적용 (Override)
                            let main = '';
                            let sub = '';
                            
                            if (truthMap.has(key)) {
                                const truth = truthMap.get(key);
                                main = truth.main;
                                sub = truth.sub;
                            } else {
                                // Fallback: 기존 로직 (그러나 신뢰도 낮음)
                                main = (s && typeof s === 'object') ? (s.main || s.main_emotion) : '';
                                main = mapMainLabel(main);
                                let subRaw = (s && typeof s === 'object') ? (s.sub || s.sub_emotion) : '';
                                sub = mapSubLabel(main, subRaw, data.raw_json && data.raw_json.sub_label_map);
                                sub = normalizeSubLabel(main, sub);
                            }
                            
                            const label = main ? `[${main}${sub && sub !== '—' ? `|${sub}` : ''}]` : '';
                            const body = text ? truncateText(text, 60) : label || JSON.stringify(s);
                            
                            return `• ${body}${label && body !== label ? ` ${label}` : ''}`;
                        });
                        return `<strong>문장 ${sentences.length}개 분석</strong><br/>${samples.join('<br/>')}`;
                    }
                    if (moduleData.relationships) {
                        return `관계 ${moduleData.relationships.length || ''}개 분석`;
                    }
                    return '';
                }
                case 'situation_analyzer': {
                    // Situation Analyzer: 상황별 분류 결과 표시
                    // ★★★ 개선: identified_situations 배열도 처리 ★★★
                    
                    const parts = [];
                    let sortedSits = [];
                    
                    // 1) identified_situations 배열 처리 (감정 기반 추론 결과 포함)
                    const identifiedSits = moduleData.identified_situations || [];
                    if (Array.isArray(identifiedSits) && identifiedSits.length > 0) {
                        sortedSits = identifiedSits
                            .map(item => ({
                                name: item.situation || item.situation_name || '',
                                score: Number(item.confidence || 0),
                                source: item.source || item.inference_source || 'matched',
                                emotion: item.primary_emotion || ''
                            }))
                            .filter(item => item.name && item.score > 0.1)
                            .sort((a, b) => b.score - a.score)
                            .slice(0, 3);
                    }
                    
                    // 2) situations 객체 처리 (기존 호환)
                    if (sortedSits.length === 0) {
                        const situations = moduleData.situations || moduleData.situation_scores;
                        if (situations && typeof situations === 'object') {
                            sortedSits = Object.entries(situations)
                                .map(([k, v]) => ({ name: k, score: Number(v), source: 'keyword' }))
                                .filter(item => !Number.isNaN(item.score) && item.score > 0.01)
                                .sort((a, b) => b.score - a.score)
                                .slice(0, 3);
                        }
                    }
                    
                    // 3) 결과 표시
                    if (sortedSits.length > 0) {
                        parts.push(`<strong>상황 분류 (Top ${sortedSits.length})</strong>`);
                        const items = sortedSits.map(item => {
                            const pct = (item.score <= 1 ? item.score * 100 : item.score).toFixed(0);
                            const sourceTag = item.source === 'emotion_inference' 
                                ? '<span style="color: #60a5fa; font-size: 0.75em;">[추론]</span>' 
                                : '';
                            return `• ${item.name} ${sourceTag}<span style="color: #9ca3af; font-size: 0.85em;">(${pct}%)</span>`;
                        });
                        parts.push(items.join('<br/>'));
                        return parts.join('<br/>');
                    }
                    
                    // 4) 요약 표시 (상황이 없을 때)
                    if (moduleData.summary && moduleData.summary.length > 10 && !moduleData.summary.includes("보편적인")) {
                         return `<strong>요약</strong><br/>${moduleData.summary}`;
                    }

                    // 상황 점수나 별도 summary가 없으면, backend module_details(summary/details)에 위임
                    return '';
                }
                case 'intensity_analyzer': {
                    const parts = [];
                    const emotionLabels = {'희': '긍정', '노': '분노', '애': '슬픔', '락': '즐거움'};
                    const levelLabels = {'high': '높음', 'medium': '중간', 'low': '낮음'};
                    
                    // ★★★ 실제 데이터 구조: moduleData.emotion_intensity 에서 추출 ★★★
                    const emotionIntensity = moduleData.emotion_intensity || {};
                    const normalizedDist = moduleData.intensity_distribution_normalized || {};
                    const globalConf = moduleData.confidence;
                    
                    const emotionKeys = ['희', '노', '애', '락'];
                    const emotionData = [];
                    
                    for (const emo of emotionKeys) {
                        // emotion_intensity에서 상세 정보 추출
                        const emoInfo = emotionIntensity[emo];
                        // intensity_distribution_normalized에서 정규화 점수 추출
                        const normScore = normalizedDist[emo];
                        
                        if (emoInfo && typeof emoInfo === 'object') {
                            const level = emoInfo.level || emoInfo.intensity_level;
                            const score = emoInfo.modified_score || emoInfo.intensity_score || normScore || 0;
                            const conf = emoInfo.confidence || globalConf;
                            emotionData.push({
                                emo,
                                label: emotionLabels[emo] || emo,
                                level: levelLabels[level] || level || '—',
                                score: typeof score === 'number' ? score : 0,
                                conf: typeof conf === 'number' ? conf : null
                            });
                        } else if (typeof normScore === 'number' && normScore > 0) {
                            // emotion_intensity가 없어도 정규화 점수가 있으면 사용
                            emotionData.push({
                                emo,
                                label: emotionLabels[emo] || emo,
                                level: normScore >= 0.5 ? '높음' : (normScore >= 0.25 ? '중간' : '낮음'),
                                score: normScore,
                                conf: globalConf
                            });
                        }
                    }
                    
                    if (emotionData.length > 0) {
                        // 점수 기준 정렬
                        emotionData.sort((a, b) => b.score - a.score);
                        const topEmotions = emotionData.slice(0, 3);
                        
                        // 지배 감정 강조
                        const dominant = topEmotions[0];
                        parts.push(`<strong>주요 감정: ${dominant.label} (${dominant.level})</strong>`);
                        
                        // 상세 점수
                        const scoreLines = topEmotions.map(e => 
                            `• ${e.label}: ${(e.score * 100).toFixed(1)}% (${e.level})`
                        );
                        parts.push(scoreLines.join('<br/>'));
                        
                        // 신뢰도
                        if (typeof globalConf === 'number') {
                            parts.push(`신뢰도: ${Math.round(globalConf * 100)}%`);
                        }
                    } else {
                        // fallback: 기존 방식
                        const products = data.products || (data.bundle && data.bundle.products) || {};
                        const p1 = products.p1 || {};
                        const intensityLabel = moduleData.intensity || p1.intensity;
                        if (intensityLabel) {
                            parts.push(`감정 강도: '${intensityLabel}'`);
                        }
                        if (typeof moduleData.confidence === 'number') {
                            parts.push(`신뢰도: ${Math.round(moduleData.confidence * 100)}%`);
                        }
                    }
                    
                    return parts.length ? parts.join('<br/>') : '';
                }
                case 'psychological_analyzer': {
                    const parts = [];
                    
                    // ★★★ 디버그: 상세 데이터 구조 확인 ★★★
                    const stabilityDetail = moduleData.stability_detail || {};
                    const maturityDetail = moduleData.maturity_detail || {};
                    const compositeScores = moduleData.composite_scores || {};
                    const cogBiases = moduleData.cognitive_biases || [];
                    
                    console.log('[psychological_analyzer] stability_detail:', stabilityDetail);
                    console.log('[psychological_analyzer] maturity_detail:', maturityDetail);
                    console.log('[psychological_analyzer] composite_scores:', compositeScores);
                    console.log('[psychological_analyzer] cognitive_biases:', cogBiases);
                    
                    // 안정성: composite_scores.stability 우선 (0~1 범위)
                    let stability = null;
                    if (typeof compositeScores.stability === 'number') {
                        stability = compositeScores.stability;
                    } else if (typeof stabilityDetail.value === 'number') {
                        stability = stabilityDetail.value;
                    } else if (typeof stabilityDetail.normalized === 'number') {
                        stability = stabilityDetail.normalized;
                    }
                    
                    console.log('[psychological_analyzer] stability final:', stability);
                    
                    if (typeof stability === 'number') {
                        const pct = stability <= 1 ? Math.round(stability * 100) : Math.round(stability);
                        parts.push(`<strong>심리 안정성 ${pct}%</strong>`);
                    }
                    
                    // 성숙도: maturity_detail (이미 위에서 선언됨)
                    const maturity = maturityDetail.score || maturityDetail.maturity || maturityDetail.value || compositeScores.maturity;
                    if (typeof maturity === 'number') {
                        const pct = maturity <= 1 ? Math.round(maturity * 100) : Math.round(maturity);
                        parts.push(`심리 성숙도: ${pct}%`);
                    }
                    
                    // 방어기제 - {mechanism: '합리화', confidence: 0.18, ...}
                    const defenses = moduleData.defense_mechanisms || [];
                    if (Array.isArray(defenses) && defenses.length > 0) {
                        const defenseItems = defenses.slice(0, 3).map(d => {
                            if (typeof d === 'string') return d;
                            if (d && typeof d === 'object') {
                                const name = d.mechanism || d.name || d.type || '';
                                const conf = d.confidence;
                                if (name && typeof conf === 'number') {
                                    return `${name}(${Math.round(conf * 100)}%)`;
                                }
                                return name;
                            }
                            return '';
                        }).filter(n => n);
                        
                        if (defenseItems.length > 0) {
                            parts.push(`방어기제: ${defenseItems.join(', ')}`);
                        }
                    }
                    
                    // 인지 편향 - cognitive_biases (이미 위에서 선언됨)
                    if (Array.isArray(cogBiases) && cogBiases.length > 0) {
                        const biasNames = cogBiases.slice(0, 2).map(b => {
                            if (typeof b === 'string') return b;
                            if (b && typeof b === 'object') {
                                return b.bias || b.name || b.type || '';
                            }
                            return '';
                        }).filter(n => n);
                        
                        if (biasNames.length > 0) {
                            parts.push(`인지 편향: ${biasNames.join(', ')}`);
                        }
                    }
                    
                    // 통찰 - insights (있으면 첫 번째 것)
                    const insights = moduleData.insights || [];
                    if (Array.isArray(insights) && insights.length > 0) {
                        const firstInsight = insights[0];
                        const insightText = typeof firstInsight === 'string' 
                            ? firstInsight 
                            : (firstInsight?.text || firstInsight?.insight || '');
                        if (insightText && insightText.length < 50) {
                            parts.push(`통찰: ${insightText}`);
                        }
                    }
                    
                    // 신뢰도
                    if (typeof moduleData.confidence === 'number') {
                        parts.push(`신뢰도: ${Math.round(moduleData.confidence * 100)}%`);
                    }
                    
                    if (parts.length > 0) {
                        return parts.join('<br/>');
                    }
                    
                    return '';
                }
                case 'complex_analyzer': {
                    // ★★★ bundle.main_dist (test.py 최종 결과) 우선 사용 ★★★
                    const bundleMainDist = (data.truth && data.truth.main_dist) 
                        || (data.bundle && data.bundle.main_dist) 
                        || data.main_distribution 
                        || null;
                    
                    console.log('[complex_analyzer] bundle.main_dist:', bundleMainDist);
                    
                    // bundle.main_dist가 있고 균등 분포가 아니면 이를 기반으로 표시
                    if (bundleMainDist && typeof bundleMainDist === 'object') {
                        const entries = Object.entries(bundleMainDist)
                            .filter(([k, v]) => typeof v === 'number' && v > 0.05)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 3);
                        
                        const values = Object.values(bundleMainDist).filter(v => typeof v === 'number');
                        const maxVal = Math.max(...values);
                        const minVal = Math.min(...values);
                        const isUniform = (maxVal - minVal) < 0.1;
                        
                        if (!isUniform && entries.length > 0) {
                            const subEmotionMap = {
                                '희': ['충만함', '자신감', '평온함'],
                                '노': ['적개심', '거부감', '분노'],
                                '애': ['상실감', '우울함', '허탈감'],
                                '락': ['즐거움', '해방감', '재미']
                            };
                            
                            const items = entries.map(([emoId, score]) => {
                                const pct = (score * 100).toFixed(1);
                                const subs = subEmotionMap[emoId] || [];
                                const topSub = subs[0] || '';
                                return topSub ? `${emoId}-${topSub} ${pct}%` : `${emoId} ${pct}%`;
                            });
                            
                            console.log('[complex_analyzer] ✅ bundle.main_dist 기반 표시:', items);
                            return `<strong>복합 감정 ${items.length}개</strong><br/>${items.map(item => `• ${item}`).join('<br/>')}`;
                        }
                    }
                    
                    // fallback: complex_analyzer 결과 사용
                    let complexData = moduleData;
                    if (data.results && data.results.complex_analyzer) {
                        complexData = data.results.complex_analyzer;
                    }
                    
                    let detectedEmotions = complexData.detected_emotions || complexData.dominant_emotions || [];
                    const entropyValue = complexData.complexity_metrics?.balance_entropy || 0;
                    const hasNoKeywords = Array.isArray(detectedEmotions) && 
                        detectedEmotions.every(e => !e.keywords || e.keywords.length === 0);
                    
                    if (entropyValue > 0.95 && hasNoKeywords) {
                        return '<span style="color:#888;">복합 감정 분석 중</span>';
                    }
                    
                    if (Array.isArray(detectedEmotions) && detectedEmotions.length > 0) {
                        detectedEmotions = [...detectedEmotions]
                            .sort((a, b) => (b.score || 0) - (a.score || 0))
                            .slice(0, 3);
                    }
                    
                    const items = formatComplexItems(detectedEmotions);
                    if (items.length) {
                        return `<strong>복합 감정 ${items.length}개</strong><br/>${items.map(item => `• ${item}`).join('<br/>')}`;
                    }
                    return '';
                }
                case 'weight_calculator': {
                    if (moduleData.features && typeof moduleData.features === 'object') {
                        const features = Object.entries(moduleData.features).slice(0, 3);
                        if (features.length) {
                            return `<strong>특징 ${features.length}개</strong><br/>${features.map(([k, v]) => `• ${k}: ${typeof v === 'number' ? v.toFixed(2) : v}`).join('<br/>')}`;
                        }
                    }
                    if (moduleData.weights) {
                        return '가중치 계산 완료';
                    }
                    return '';
                }
                default:
                    return '';
            }
        };

        orderedNames.forEach((name, index) => {
            const displayInfo = MODULE_DISPLAY_INFO[name] || { label: `${name}`, desc: '' };
            const fallbackKey = MODULE_RESULT_FALLBACK[name];
            const moduleData = (moduleResults[name] && typeof moduleResults[name] === 'object')
                ? moduleResults[name]
                : (fallbackKey && moduleResults[fallbackKey] && typeof moduleResults[fallbackKey] === 'object'
                    ? moduleResults[fallbackKey]
                    : {});
            const detailInfo = moduleDetailMap.get(name) || {};
            const status = detailInfo.status || (moduleHitRate[name] ? 'ok' : 'missing');
            const block = document.createElement('div');
            block.className = 'module-block';
            block.style.padding = '12px';
            block.style.backgroundColor = '#1a1a2e';
            block.style.borderRadius = '8px';
            block.style.border = '1px solid #323244';

            const title = document.createElement('div');
            title.style.fontSize = '13px';
            title.style.fontWeight = '600';
            title.style.color = '#e0e7ff';
            title.style.marginBottom = '4px';
            title.textContent = `${index + 1}. ${displayInfo.label || name}`;

            const badge = document.createElement('span');
            badge.style.marginLeft = '8px';
            badge.style.fontSize = '11px';
            badge.style.padding = '2px 6px';
            badge.style.borderRadius = '999px';
            badge.style.background = MODULE_STATUS_COLORS[status] || '#4b5563';
            badge.style.color = '#0f172a';
            badge.style.fontWeight = '600';
            badge.textContent = MODULE_STATUS_LABEL[status] || status.toUpperCase();
            title.appendChild(badge);
            block.appendChild(title);

            if (displayInfo.desc) {
                const descEl = document.createElement('div');
                descEl.style.fontSize = '11px';
                descEl.style.color = '#9999aa';
                descEl.style.marginBottom = '8px';
                descEl.textContent = displayInfo.desc;
                block.appendChild(descEl);
            }

            const result = document.createElement('div');
            result.style.fontSize = '12px';
            result.style.color = '#e5e5e5';
            result.style.lineHeight = '1.4';

            let detailHTML = '';
            if (status === 'ok') {
                detailHTML = buildModuleInsight(name, moduleData);
            }
            // ★★★ summary와 details를 함께 표시 ★★★
            if (!detailHTML) {
                const parts = [];
                // summary가 있으면 먼저 표시 (강조)
                if (detailInfo.summary && detailInfo.summary.trim()) {
                    parts.push(`<strong>${detailInfo.summary}</strong>`);
                }
                // details가 있으면 bullet point로 표시
                if (Array.isArray(detailInfo.details) && detailInfo.details.length) {
                    parts.push(detailInfo.details.map(line => `• ${line}`).join('<br/>'));
                }
                if (parts.length) {
                    detailHTML = parts.join('<br/>');
                }
            }
            if (!detailHTML) {
                detailHTML = status === 'ok'
                    ? '모듈이 정상 실행되었습니다.'
                    : '모듈 실행 데이터가 보고되지 않았습니다.';
            }

            result.innerHTML = detailHTML;
            block.appendChild(result);
            grid.appendChild(block);
        });

        container.appendChild(grid);
    },

    renderMasterReport(data) {
        const container = this.getEl('analysisMasterReport');
        if (!container) return;

        const report = data.master_report || '';
        const header = container.previousElementSibling;
        const existing = document.getElementById('analysisMasterReportCollapsible');
        if (existing) {
            existing.remove();
        }

        if (!report) {
            container.textContent = '마스터 리포트가 없습니다.';
            container.style.display = '';
            if (header) header.style.display = '';
            return;
        }

        container.style.display = 'none';
        if (header) {
            header.style.display = '';
        }

        const wrapper = document.createElement('details');
        wrapper.id = 'analysisMasterReportCollapsible';
        wrapper.className = 'collapsed-report';

        const summary = document.createElement('summary');
        summary.textContent = '마스터 리포트 전문 열기';

        const pre = document.createElement('pre');
        pre.className = 'explain-pre';
        pre.textContent = report;

        wrapper.appendChild(summary);
        wrapper.appendChild(pre);

        container.parentElement.insertBefore(wrapper, container.nextSibling);
    }
};

// 전역으로 노출 (하위 호환성 유지)
window.AnalysisRenderer = AnalysisRenderer;

// ============================================
// ExpertViewRenderer: Truth 필드 원본 표시 전용 렌더러
// ============================================
const ExpertViewRenderer = {
    /**
     * Expert View 렌더링 - data.truth만 사용하여 Truth 필드를 그대로 표시
     * @param {Object} data - alignResultData()로 정렬된 데이터 (data.truth 포함)
     */
    render(data) {
        const block = document.getElementById('expertViewBlock');
        const content = document.getElementById('expertViewContent');
        const fieldsContainer = document.getElementById('expertViewTruthFields');
        
        if (!block || !content || !fieldsContainer) return;
        
        // data.truth가 없으면 Expert View 숨김
        if (!data.truth || typeof data.truth !== 'object') {
            block.style.display = 'none';
            return;
        }
        
        // Expert View 표시
        block.style.display = 'block';
        
        // Truth 필드 렌더링
        fieldsContainer.innerHTML = '';
        
        // 1. main_dist (메인 감정 분포)
        this.renderMainDist(data.truth.main_dist, fieldsContainer);
        
        // 2. sub_top / sub_top10_lines (세부 감정)
        this.renderSubTop(data.truth.sub_top, data.truth.sub_top10_lines, fieldsContainer);
        
        // 3. sentence_annotations_structured (문장별 감정 태깅)
        this.renderSentenceAnnotations(data.truth.sentence_annotations_structured, fieldsContainer);
        
        // 4. transitions_structured (감정 전이 구조)
        this.renderTransitions(data.truth.transitions_structured, fieldsContainer);
        
        // 5. why_lines (왜 이런 감정인가)
        this.renderWhyLines(data.truth.why_lines, fieldsContainer);
        
        // 6. reasoning_path_lines (추론 경로)
        this.renderReasoningPath(data.truth.reasoning_path_lines, fieldsContainer);
        
        // 7. flow_ssot (감정 흐름 요약)
        this.renderFlowSSOT(data.truth.flow_ssot, fieldsContainer);
        
        // 8. triggers (트리거/키워드)
        this.renderTriggers(data.truth.triggers, fieldsContainer);
        
        // 9. products (제품/리포트)
        this.renderProducts(data.truth.products, fieldsContainer);
        
        // 10. reports (CS/BI 리포트)
        this.renderReports(data.truth.reports, fieldsContainer);
        
        // 11. meta (메타 정보)
        this.renderMeta(data.truth.meta, fieldsContainer);
        
        // 12. RAW JSON 다운로드 버튼
        this.renderRawJsonDownload(data.bundle || data.truth, fieldsContainer);
    },
    
    /**
     * 섹션 헤더 생성
     */
    createSectionHeader(title, description = '') {
        const header = document.createElement('div');
        header.className = 'expert-section-header';
        header.innerHTML = `
            <h3 class="expert-section-title">${title}</h3>
            ${description ? `<p class="expert-section-desc">${description}</p>` : ''}
        `;
        return header;
    },
    
    /**
     * 필드 카드 생성
     */
    createFieldCard(title, content, source = '') {
        const card = document.createElement('div');
        card.className = 'expert-field-card';
        card.innerHTML = `
            <div class="expert-field-header">
                <span class="expert-field-title">${title}</span>
                ${source ? `<span class="expert-field-source">source: ${source}</span>` : ''}
            </div>
            <div class="expert-field-content">${content}</div>
        `;
        return card;
    },
    
    /**
     * main_dist 렌더링
     */
    renderMainDist(mainDist, container) {
        if (!mainDist || typeof mainDist !== 'object') return;
        
        const section = document.createElement('div');
        section.className = 'expert-section';
        section.appendChild(this.createSectionHeader('main_dist', '메인 감정 분포 (희/노/애/락 → 0~1)'));
        
        const content = Object.entries(mainDist)
            .map(([emotion, value]) => {
                const pct = typeof value === 'number' ? (value * 100).toFixed(2) : '—';
                return `<div class="expert-dist-item"><strong>${emotion}</strong>: ${value} (${pct}%)</div>`;
            })
            .join('');
        
        section.appendChild(this.createFieldCard('메인 감정 분포', content, 'data.truth.main_dist'));
        container.appendChild(section);
    },
    
    /**
     * sub_top / sub_top10_lines 렌더링
     */
    renderSubTop(subTop, subTop10Lines, container) {
        const section = document.createElement('div');
        section.className = 'expert-section';
        section.appendChild(this.createSectionHeader('sub_top / sub_top10_lines', '세부 감정 Rank (sub label, p=0~100 퍼센트)'));
        
        let content = '';
        
        if (Array.isArray(subTop) && subTop.length > 0) {
            content += '<div class="expert-sub-top"><strong>sub_top (배열):</strong><pre>' + 
                JSON.stringify(subTop, null, 2) + '</pre></div>';
        }
        
        if (Array.isArray(subTop10Lines) && subTop10Lines.length > 0) {
            content += '<div class="expert-sub-top10"><strong>sub_top10_lines (라인 형식):</strong><pre>' + 
                subTop10Lines.join('\n') + '</pre></div>';
        }
        
        if (!content) {
            content = '<div class="expert-empty">세부 감정 정보 없음</div>';
        }
        
        section.appendChild(this.createFieldCard('세부 감정', content, 'data.truth.sub_top / data.truth.sub_top10_lines'));
        container.appendChild(section);
    },
    
    /**
     * sentence_annotations_structured 렌더링
     */
    renderSentenceAnnotations(annotations, container) {
        if (!Array.isArray(annotations) || annotations.length === 0) return;
        
        const section = document.createElement('div');
        section.className = 'expert-section';
        section.appendChild(this.createSectionHeader('sentence_annotations_structured', '문장별 감정 태깅'));
        
        const content = '<pre>' + JSON.stringify(annotations, null, 2) + '</pre>';
        section.appendChild(this.createFieldCard('문장별 감정 태깅', content, 'data.truth.sentence_annotations_structured'));
        container.appendChild(section);
    },
    
    /**
     * transitions_structured 렌더링
     */
    renderTransitions(transitions, container) {
        if (!Array.isArray(transitions) || transitions.length === 0) return;
        
        const section = document.createElement('div');
        section.className = 'expert-section';
        section.appendChild(this.createSectionHeader('transitions_structured', '감정 전이 구조'));
        
        const content = '<pre>' + JSON.stringify(transitions, null, 2) + '</pre>';
        section.appendChild(this.createFieldCard('감정 전이 구조', content, 'data.truth.transitions_structured'));
        container.appendChild(section);
    },
    
    /**
     * why_lines 렌더링
     */
    renderWhyLines(whyLines, container) {
        if (!Array.isArray(whyLines) || whyLines.length === 0) return;
        
        const section = document.createElement('div');
        section.className = 'expert-section';
        section.appendChild(this.createSectionHeader('why_lines', '왜 이런 감정인가 설명'));
        
        const content = '<pre>' + whyLines.join('\n') + '</pre>';
        section.appendChild(this.createFieldCard('왜 이런 감정인가', content, 'data.truth.why_lines'));
        container.appendChild(section);
    },
    
    /**
     * reasoning_path_lines 렌더링
     */
    renderReasoningPath(reasoningPathLines, container) {
        if (!Array.isArray(reasoningPathLines) || reasoningPathLines.length === 0) return;
        
        const section = document.createElement('div');
        section.className = 'expert-section';
        section.appendChild(this.createSectionHeader('reasoning_path_lines', '추론 경로 단계 설명'));
        
        const content = '<pre>' + reasoningPathLines.join('\n') + '</pre>';
        section.appendChild(this.createFieldCard('추론 경로', content, 'data.truth.reasoning_path_lines'));
        container.appendChild(section);
    },
    
    /**
     * flow_ssot 렌더링
     */
    renderFlowSSOT(flowSSOT, container) {
        if (!flowSSOT) return;
        
        const section = document.createElement('div');
        section.className = 'expert-section';
        section.appendChild(this.createSectionHeader('flow_ssot', '감정 흐름 요약 (SSOT)'));
        
        const content = typeof flowSSOT === 'string' 
            ? '<pre>' + flowSSOT + '</pre>'
            : '<pre>' + JSON.stringify(flowSSOT, null, 2) + '</pre>';
        section.appendChild(this.createFieldCard('감정 흐름 요약', content, 'data.truth.flow_ssot'));
        container.appendChild(section);
    },
    
    /**
     * triggers 렌더링
     */
    renderTriggers(triggers, container) {
        if (!triggers) return;
        
        const section = document.createElement('div');
        section.className = 'expert-section';
        section.appendChild(this.createSectionHeader('triggers', '트리거/키워드'));
        
        const content = typeof triggers === 'string'
            ? '<pre>' + triggers + '</pre>'
            : '<pre>' + JSON.stringify(triggers, null, 2) + '</pre>';
        section.appendChild(this.createFieldCard('트리거/키워드', content, 'data.truth.triggers'));
        container.appendChild(section);
    },
    
    /**
     * products 렌더링
     */
    renderProducts(products, container) {
        if (!products || typeof products !== 'object') return;
        
        const section = document.createElement('div');
        section.className = 'expert-section';
        section.appendChild(this.createSectionHeader('products', '제품/리포트 (p1/p3/p5)'));
        
        const content = '<pre>' + JSON.stringify(products, null, 2) + '</pre>';
        section.appendChild(this.createFieldCard('제품/리포트', content, 'data.truth.products'));
        container.appendChild(section);
    },
    
    /**
     * reports 렌더링
     */
    renderReports(reports, container) {
        if (!reports || typeof reports !== 'object') return;
        
        const section = document.createElement('div');
        section.className = 'expert-section';
        section.appendChild(this.createSectionHeader('reports', 'CS/BI 리포트 요약'));
        
        const content = '<pre>' + JSON.stringify(reports, null, 2) + '</pre>';
        section.appendChild(this.createFieldCard('CS/BI 리포트', content, 'data.truth.reports'));
        container.appendChild(section);
    },
    
    /**
     * meta 렌더링
     */
    renderMeta(meta, container) {
        if (!meta || typeof meta !== 'object') return;
        
        const section = document.createElement('div');
        section.className = 'expert-section';
        section.appendChild(this.createSectionHeader('meta', '메타 정보 (evidence_score, evidence_label 등)'));
        
        const content = '<pre>' + JSON.stringify(meta, null, 2) + '</pre>';
        section.appendChild(this.createFieldCard('메타 정보', content, 'data.truth.meta'));
        container.appendChild(section);
    },
    
    /**
     * RAW JSON 다운로드 버튼 렌더링
     */
    renderRawJsonDownload(bundle, container) {
        if (!bundle || typeof bundle !== 'object') return;
        
        const section = document.createElement('div');
        section.className = 'expert-section';
        section.appendChild(this.createSectionHeader('RAW JSON', '전체 bundle 원본 다운로드'));
        
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'expert-download-btn';
        downloadBtn.textContent = '📥 bundle.json 다운로드';
        downloadBtn.onclick = () => {
            const jsonStr = JSON.stringify(bundle, null, 2);
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `bundle_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        };
        
        const card = document.createElement('div');
        card.className = 'expert-field-card';
        card.appendChild(downloadBtn);
        section.appendChild(card);
        container.appendChild(section);
    }
};

// AnalysisRenderer에 Expert View 렌더링 메서드 추가
AnalysisRenderer.renderExpertView = function(data) {
    ExpertViewRenderer.render(data);
};

// Expert View 토글 이벤트 리스너
document.addEventListener('DOMContentLoaded', function() {
    const toggleBtn = document.getElementById('expertViewToggle');
    const content = document.getElementById('expertViewContent');
    
    if (toggleBtn && content) {
        toggleBtn.addEventListener('click', function() {
            const isHidden = content.style.display === 'none';
            content.style.display = isHidden ? 'block' : 'none';
            toggleBtn.querySelector('svg').style.transform = isHidden ? 'rotate(180deg)' : 'rotate(0deg)';
        });
    }
});

const VideoController = {
    init() {
        const video = document.querySelector('.main-hero .video');
        if (video) video.playbackRate = 0.7;
        
        const middleVideo = document.querySelector('.main-item1 .wrapper .middle .middle-image');
        const playBtn = document.querySelector('.main-item1 .wrapper .middle .video-play-btn');
        
        if (middleVideo && playBtn) {
            playBtn.addEventListener('click', () => {
                middleVideo.play();
                playBtn.classList.add('hidden');
            });
            
            middleVideo.addEventListener('play', () => {
                playBtn.classList.add('hidden');
            });
            
            middleVideo.addEventListener('pause', () => {
                playBtn.classList.remove('hidden');
            });
        }
        
        const cardVideos = document.querySelectorAll('.main-item1 .bottom .card-media video, .main-item2 .wrapper .bottom .card-media video, .main-item3 .bottom .card-media video');
        
        cardVideos.forEach((video) => {
            const cardMedia = video.closest('.card-media');
            const playBtn = cardMedia?.querySelector('.video-play-btn');
            const pauseBtn = cardMedia?.querySelector('.video-pause-btn');
            
            if (playBtn && pauseBtn) {
                playBtn.addEventListener('click', () => {
                    video.play();
                    playBtn.classList.add('hidden');
                    pauseBtn.classList.add('active');
                });
                
                pauseBtn.addEventListener('click', () => {
                    video.pause();
                    pauseBtn.classList.remove('active');
                    playBtn.classList.remove('hidden');
                });
                
                video.addEventListener('play', () => {
                    playBtn.classList.add('hidden');
                    pauseBtn.classList.add('active');
                });
                
                video.addEventListener('pause', () => {
                    pauseBtn.classList.remove('active');
                    playBtn.classList.remove('hidden');
                });
            }
        });
    }
};

const Slider = {
    create(config) {
        const { selector, type = 'image' } = config;
        
        if (type === 'image') {
            return this.createImageSlider(selector);
        } else if (type === 'card') {
            return this.createCardSlider(selector);
        }
    },
    
    createImageSlider(selector) {
        const slider = document.querySelector(selector);
        if (!slider) {
            console.warn(`[Slider] 슬라이더를 찾을 수 없습니다: ${selector}`);
            return;
        }
        
        const sliderContainer = slider.querySelector('.slider-container');
        if (!sliderContainer) {
            console.warn(`[Slider] .slider-container를 찾을 수 없습니다: ${selector}`);
            return;
        }
        
        // slider-track 내부에서 slides 찾기 (더 정확한 선택)
        const sliderTrack = sliderContainer.querySelector('.slider-track');
        const slides = sliderTrack 
            ? sliderTrack.querySelectorAll('.slider-slide')
            : sliderContainer.querySelectorAll('.slider-slide');
        
        const sliderNav = sliderContainer.querySelector('.slider-nav');
        if (!sliderNav) {
            console.warn(`[Slider] .slider-nav를 찾을 수 없습니다: ${selector}`);
            return;
        }
        
        const dots = sliderNav.querySelectorAll('.slider-dot');
        const prevBtn = sliderNav.querySelector('.slider-prev');
        const nextBtn = sliderNav.querySelector('.slider-next');
        
        if (!slides.length) {
            console.warn(`[Slider] 슬라이드가 없습니다: ${selector}`);
            return;
        }
        
        if (!dots.length) {
            console.warn(`[Slider] 슬라이드 도트가 없습니다: ${selector}`);
            return;
        }
        
        let currentSlide = 0;
        const totalSlides = slides.length;
        const AUTOPLAY_DELAY = 4000; // 4초로 조정
        let autoplayTimer = null;
        let isTransitioning = false;
        
        const showSlide = (index) => {
            if (isTransitioning) return;
            if (index < 0 || index >= totalSlides) return;
            
            isTransitioning = true;
            
            // 모든 슬라이드에서 active 제거
            slides.forEach((slide, i) => {
                slide.classList.remove('active');
                if (i === index) {
                    // 약간의 지연 후 active 추가 (CSS transition을 위해)
                    requestAnimationFrame(() => {
                        slide.classList.add('active');
                    });
                }
            });
            
            // 모든 도트에서 active 제거
            dots.forEach((dot, i) => {
                dot.classList.toggle('active', i === index);
            });
            
            currentSlide = index;
            
            // 전환 완료
            setTimeout(() => {
                isTransitioning = false;
            }, 300);
        };
        
        const startAutoplay = () => {
            if (totalSlides <= 1) return;
            stopAutoplay();
            autoplayTimer = setInterval(() => {
                if (!isTransitioning) {
                    nextSlide(false);
                }
            }, AUTOPLAY_DELAY);
        };
        
        const stopAutoplay = () => {
            if (autoplayTimer) {
                clearInterval(autoplayTimer);
                autoplayTimer = null;
            }
        };
        
        const resetAutoplay = () => {
            stopAutoplay();
            startAutoplay();
        };
        
        const nextSlide = (shouldReset = true) => {
            if (isTransitioning) return;
            const next = (currentSlide + 1) % totalSlides;
            showSlide(next);
            if (shouldReset) resetAutoplay();
        };
        
        const prevSlide = () => {
            if (isTransitioning) return;
            const prev = (currentSlide - 1 + totalSlides) % totalSlides;
            showSlide(prev);
            resetAutoplay();
        };
        
        // 이벤트 리스너 등록
        if (nextBtn) {
            nextBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (!isTransitioning) {
                    nextSlide();
                }
            });
        } else {
            console.warn(`[Slider] 다음 버튼을 찾을 수 없습니다: ${selector}`);
        }
        
        if (prevBtn) {
            prevBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (!isTransitioning) {
                    prevSlide();
                }
            });
        } else {
            console.warn(`[Slider] 이전 버튼을 찾을 수 없습니다: ${selector}`);
        }
        
        dots.forEach((dot, index) => {
            dot.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (!isTransitioning && index !== currentSlide) {
                    showSlide(index);
                    resetAutoplay();
                }
            });
        });
        
        // 초기화
        showSlide(0);
        startAutoplay();
        
        // 마우스 호버 시 autoplay 일시정지
        slider.addEventListener('mouseenter', stopAutoplay);
        slider.addEventListener('mouseleave', startAutoplay);
        
        console.log(`[Slider] 이미지 슬라이더 초기화 완료: ${selector} (${totalSlides}개 슬라이드)`);
    },
    
    createCardSlider(selector) {
        const cardSlider = document.querySelector(selector);
        if (!cardSlider) return;
        
        const sliderWrapper = cardSlider.querySelector('.slider-wrapper');
        const sliderTrack = cardSlider.querySelector('.slider-track');
        const sliderCards = cardSlider.querySelectorAll('.slider-card');
        const prevBtn = cardSlider.querySelector('.slider-nav-btn.prev');
        const nextBtn = cardSlider.querySelector('.slider-nav-btn.next');
        
        if (!sliderWrapper || !sliderTrack || !sliderCards.length) return;
        
        let currentIndex = 0;
        const totalCards = sliderCards.length;
        let resizeTimeout = null;
        
        // 반응형 gap 계산 함수
        const getGap = () => {
            const width = window.innerWidth;
            if (width <= 800) {
                return 0;
            } else if (width <= 1024) {
                return 24; // 1.5rem = 24px (2개 슬라이드)
            }
            return 24; // 1.5rem = 24px (3개 슬라이드)
        };
        
        // 반응형 cardsPerView 계산 함수
        const getCardsPerView = () => {
            const width = window.innerWidth;
            if (width <= 800) {
                return 1;
            } else if (width <= 1024) {
                return 2;
            }
            return 3;
        };
        
        let cardsPerView = getCardsPerView();
        
        const getCardWidth = () => {
            const containerWidth = sliderWrapper.offsetWidth;
            if (containerWidth === 0) {
                // 아직 렌더링되지 않았으면 기본값 반환
                return 300;
            }
            const currentCardsPerView = getCardsPerView();
            const currentGap = getGap();
            return (containerWidth - (currentGap * (currentCardsPerView - 1))) / currentCardsPerView;
        };
        
        const updateVisibleCards = () => {
            const currentCardsPerView = getCardsPerView();
            sliderCards.forEach((card, index) => {
                // 현재 보이는 범위의 슬라이드만 visible로 설정
                const isVisible = index >= currentIndex && index < currentIndex + currentCardsPerView;
                if (isVisible) {
                    card.classList.add('visible');
                } else {
                    card.classList.remove('visible');
                }
                card.classList.toggle('active', index === currentIndex);
            });
        };
        
        const updateSlider = () => {
            const currentCardsPerView = getCardsPerView();
            const currentGap = getGap();
            const cardWidth = getCardWidth();
            
            const translateX = -(currentIndex * (cardWidth + currentGap));
            sliderTrack.style.transform = `translateX(${translateX}px)`;
            sliderTrack.style.transition = 'transform 0.3s ease';
            
            updateVisibleCards();
            
            // 현재 인덱스가 유효한 범위를 벗어나면 조정
            const maxIndex = Math.max(0, totalCards - currentCardsPerView);
            if (currentIndex > maxIndex) {
                currentIndex = maxIndex;
            }
            
            // loop 기능이므로 버튼은 항상 활성화
            if (prevBtn) {
                prevBtn.disabled = false;
                prevBtn.classList.remove('disabled');
            }
            if (nextBtn) {
                nextBtn.disabled = false;
                nextBtn.classList.remove('disabled');
            }
            
            // 3개 슬라이드일 때만 그라데이션 효과 표시 (1024px 초과)
            if (currentCardsPerView === 3) {
                // 3개 슬라이드일 때는 has-prev, has-next 클래스로 그라데이션 제어
                if (currentIndex === 0) {
                    cardSlider.classList.remove('has-prev');
                } else {
                    cardSlider.classList.add('has-prev');
                }
                
                if (currentIndex >= maxIndex) {
                    cardSlider.classList.remove('has-next');
                } else {
                    cardSlider.classList.add('has-next');
                }
            } else {
                // 2개 이하 슬라이드일 때는 그라데이션 효과 제거
                cardSlider.classList.remove('has-prev', 'has-next');
            }
        };
        
        const nextSlide = () => {
            const currentCardsPerView = getCardsPerView();
            const maxIndex = Math.max(0, totalCards - currentCardsPerView);
            
            if (currentIndex < maxIndex) {
                currentIndex++;
            } else {
                // 마지막 슬라이드에서 첫 번째로 loop
                currentIndex = 0;
            }
            updateSlider();
        };
        
        const prevSlide = () => {
            const currentCardsPerView = getCardsPerView();
            const maxIndex = Math.max(0, totalCards - currentCardsPerView);
            
            if (currentIndex > 0) {
                currentIndex--;
            } else {
                // 첫 번째 슬라이드에서 마지막으로 loop
                currentIndex = maxIndex;
            }
            updateSlider();
        };
        
        // 리사이즈 핸들러 (디바운싱)
        const handleResize = () => {
            if (resizeTimeout) {
                clearTimeout(resizeTimeout);
            }
            resizeTimeout = setTimeout(() => {
                const oldCardsPerView = cardsPerView;
                cardsPerView = getCardsPerView();
                
                // 화면 크기가 변경되면 인덱스 조정
                const maxIndex = Math.max(0, totalCards - cardsPerView);
                if (currentIndex > maxIndex) {
                    currentIndex = maxIndex;
                }
                
                updateSlider();
            }, 150);
        };
        
        // 이벤트 리스너 등록
        if (nextBtn) {
            nextBtn.addEventListener('click', nextSlide);
        }
        
        if (prevBtn) {
            prevBtn.addEventListener('click', prevSlide);
        }
        
        // 리사이즈 이벤트 등록
        window.addEventListener('resize', handleResize);
        
        // 초기화: 약간의 지연을 두고 실행 (DOM이 완전히 렌더링된 후)
        // requestAnimationFrame을 사용하여 브라우저 렌더링 사이클과 동기화
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                // 초기 인덱스 설정
                currentIndex = 0;
                
                // 슬라이더 트랙 초기 위치 설정
                sliderTrack.style.transform = 'translateX(0px)';
                sliderTrack.style.transition = 'none';
                
                // 짧은 지연 후 transition 활성화 및 업데이트
                setTimeout(() => {
                    sliderTrack.style.transition = 'transform 0.3s ease';
                    updateSlider();
                }, 50);
            });
        });
    },
    
    init() {
        // 이미지 슬라이더 초기화 (.slider 클래스)
        const allSliders = document.querySelectorAll('.slider');
        allSliders.forEach(slider => {
            const type = slider.dataset.sliderType || 'image';
            // [FIX] Class selection robustness
            // slider.classList[0] might not be the unique identifier.
            // Use the element directly if possible or a more specific class.
            // If the first class is 'slider', we need to find another one or use ID.
            let selector = `.${slider.classList[0]}`;
            if (slider.id) {
                selector = `#${slider.id}`;
            } else if (slider.classList.length > 1) {
                 // Find a class that is not 'slider'
                 const uniqueClass = Array.from(slider.classList).find(c => c !== 'slider');
                 if (uniqueClass) selector = `.${uniqueClass}`;
            }
            
            this.create({ selector, type });
        });
        
        // top-image-slider 초기화 (main-item3)
        const topImageSlider = document.querySelector('.top-image-slider');
        if (topImageSlider) {
            this.createImageSlider('.top-image-slider');
        }
        
        // 카드 슬라이더 초기화 (main-item4)
        const cardSlider = document.querySelector('.main-item4 .card-slider');
        if (cardSlider) {
            this.createCardSlider('.main-item4 .card-slider');
        }
    }
};

const Modal = {
    init() {
        const closeMobileMenuIfOpen = () => {
            const mobileMenu = document.getElementById('mobileMenu');
            if (mobileMenu && mobileMenu.classList.contains('active')) {
                mobileMenu.classList.remove('active');
                mobileMenu.setAttribute('aria-hidden', 'true');
                const toggleButton = document.querySelector('.menu-toggle');
                toggleButton?.setAttribute('aria-expanded', 'false');
            }
        };
        
        // data-modal 속성을 가진 요소들 처리 (feature-item 포함)
        document.querySelectorAll('[data-modal]').forEach(trigger => {
            trigger.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const targetId = trigger.getAttribute('data-modal');
                const targetModal = targetId ? document.getElementById(targetId) : null;
                if (!targetModal) {
                    console.warn('[Modal] 모달을 찾을 수 없습니다:', targetId);
                    return;
                }
                
                closeMobileMenuIfOpen();
                const iframe = targetModal.querySelector('.ir-modal-viewer iframe[data-src]');
                if (iframe && !iframe.src) {
                    iframe.src = iframe.dataset.src;
                }
                targetModal.classList.add('active');
                targetModal.setAttribute('aria-hidden', 'false');
                document.body.style.overflow = 'hidden';
            });
            
            // feature-item에 포인터 커서 추가
            if (trigger.classList.contains('feature-item')) {
                trigger.style.cursor = 'pointer';
            }
        });
        
        // feature-modal 닫기 버튼 처리
        document.querySelectorAll('.feature-modal-overlay .modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const overlay = btn.closest('.feature-modal-overlay');
                if (overlay) {
                    overlay.classList.remove('active');
                    overlay.setAttribute('aria-hidden', 'true');
                    document.body.style.overflow = '';
                }
            });
        });
        
        // feature-modal 오버레이 클릭 시 닫기
        document.querySelectorAll('.feature-modal-overlay').forEach(overlay => {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    overlay.classList.remove('active');
                    overlay.setAttribute('aria-hidden', 'true');
                    document.body.style.overflow = '';
                }
            });
        });
        
        // ESC 키로 모달 닫기
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                document.querySelectorAll('.feature-modal-overlay.active').forEach(overlay => {
                    overlay.classList.remove('active');
                    overlay.setAttribute('aria-hidden', 'true');
                    document.body.style.overflow = '';
                });
                // modalOverlay도 닫기
                const modalOverlay = document.getElementById('modalOverlay');
                if (modalOverlay && modalOverlay.classList.contains('active')) {
                    modalOverlay.classList.remove('active');
                    document.body.style.overflow = '';
                }
            }
        });
        
        // modalOverlay 닫기 버튼 처리
        const modalCloseBtn = document.getElementById('modalClose');
        if (modalCloseBtn) {
            modalCloseBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const modalOverlay = document.getElementById('modalOverlay');
                if (modalOverlay) {
                    modalOverlay.classList.remove('active');
                    document.body.style.overflow = '';
                }
            });
        }
        
        // modalOverlay 오버레이 클릭 시 닫기
        const modalOverlay = document.getElementById('modalOverlay');
        if (modalOverlay) {
            modalOverlay.addEventListener('click', (e) => {
                if (e.target === modalOverlay) {
                    modalOverlay.classList.remove('active');
                    document.body.style.overflow = '';
                }
            });
        }
        
        // modules-btn 클릭 처리 (모달 열기 또는 스크롤)
        const modulesBtn = document.getElementById('modules-btn');
        if (modulesBtn) {
            modulesBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();

                const modulesModal = document.getElementById('modulesModal');
                if (modulesModal) {
                    modulesModal.classList.add('active');
                    document.body.style.overflow = 'hidden';
                    return;
                }

                const targetModal = document.getElementById('modalOverlay');
                if (targetModal) {
                    targetModal.classList.add('active');
                    document.body.style.overflow = 'hidden';
                } else {
                    const modulesSection = document.getElementById('modules');
                    if (modulesSection) {
                        modulesSection.scrollIntoView({ behavior: 'smooth' });
                    }
                }
            });
        }
        
        // pipeline-btn 클릭 처리 (필요시 모달 연결)
        const pipelineBtn = document.getElementById('pipeline-btn');
        if (pipelineBtn) {
            pipelineBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                // 파이프라인 모달이 있으면 열기, 없으면 기본 동작
                const pipelineModal = document.getElementById('pipelineModal');
                if (pipelineModal) {
                    pipelineModal.classList.add('active');
                    document.body.style.overflow = 'hidden';
                } else {
                    console.log('[Modal] 파이프라인 모달이 없습니다.');
                }
            });
        }
    }
};

const Tabs = {
    init() {
        const tabItems = Array.from(document.querySelectorAll('.main-item2 .wrapper .middle .tab-item'));
        if (!tabItems.length) return;

        const getPane = (tabItem) => {
            const targetId = tabItem?.getAttribute('data-tab');
            return targetId ? document.getElementById(targetId) : null;
        };

        let activeIndex = tabItems.findIndex(item => item.classList.contains('active'));
        if (activeIndex === -1) {
            activeIndex = 0;
            tabItems[0].classList.add('active');
            const initialPane = getPane(tabItems[0]);
            if (initialPane) initialPane.classList.add('active');
        }

        // 전환 중 상태 관리 - 더 강력한 버전
        let isTransitioning = false;
        let transitionTimeouts = new Set();
        let transitionEndHandlers = new Map();

        // 모든 진행 중인 전환 정리
        const cancelAllTransitions = () => {
            // 모든 타이머 취소
            transitionTimeouts.forEach(timeout => clearTimeout(timeout));
            transitionTimeouts.clear();
            
            // 모든 이벤트 리스너 제거
            transitionEndHandlers.forEach((handler, pane) => {
                if (pane && handler) {
                    pane.removeEventListener('transitionend', handler);
                }
            });
            transitionEndHandlers.clear();
            
            // 모든 패널에서 is-leaving 제거 (하지만 active는 유지)
            tabItems.forEach(item => {
                const pane = getPane(item);
                if (pane) {
                    pane.classList.remove('is-leaving');
                }
            });
        };

        const cleanupPane = (pane) => {
            if (!pane) return;
            // 이벤트 리스너 제거
            const handler = transitionEndHandlers.get(pane);
            if (handler) {
                pane.removeEventListener('transitionend', handler);
                transitionEndHandlers.delete(pane);
            }
            pane.classList.remove('active', 'is-leaving');
        };

        const switchTab = (targetIndex) => {
            // 같은 탭이면 무시
            if (targetIndex === activeIndex) return;

            const currentItem = tabItems[activeIndex];
            const currentPane = getPane(currentItem);
            const nextItem = tabItems[targetIndex];
            const nextPane = getPane(nextItem);

            if (!nextPane) return;

            // 전환 중이면 모든 진행 중인 전환 취소하고 새 전환 시작
            if (isTransitioning) {
                cancelAllTransitions();
            }

            // 전환 시작
            isTransitioning = true;
            activeIndex = targetIndex; // 즉시 업데이트하여 중복 클릭 방지

            // 모든 탭 아이템에서 active 제거
            tabItems.forEach(tab => tab.classList.remove('active'));
            nextItem.classList.add('active');

            // 현재 패널 숨기기
            if (currentPane && currentPane.classList.contains('active')) {
                currentPane.classList.add('is-leaving');
                
                // transitionend 이벤트 핸들러
                const handleTransitionEnd = (event) => {
                    // 이벤트가 현재 패널에서 발생했고, activeIndex가 여전히 targetIndex인지 확인
                    if (event.target !== currentPane || activeIndex !== targetIndex) {
                        return;
                    }
                    cleanupPane(currentPane);
                    isTransitioning = false;
                };
                
                transitionEndHandlers.set(currentPane, handleTransitionEnd);
                currentPane.addEventListener('transitionend', handleTransitionEnd, { once: true });
                
                // 안전장치: 최대 700ms 후 강제 정리
                const timeout = setTimeout(() => {
                    if (activeIndex === targetIndex) {
                        cleanupPane(currentPane);
                        isTransitioning = false;
                    }
                    transitionTimeouts.delete(timeout);
                }, 700);
                transitionTimeouts.add(timeout);
            } else {
                // 현재 패널이 없으면 즉시 전환
                isTransitioning = false;
            }

            // 다음 패널 표시
            nextPane.classList.remove('is-leaving');
            // 약간의 지연 후 active 추가 (CSS transition을 위해)
            requestAnimationFrame(() => {
                // activeIndex가 여전히 targetIndex인지 확인 (다른 전환이 시작되었을 수 있음)
                if (activeIndex === targetIndex) {
                    nextPane.classList.add('active');
                    // 전환 완료 확인을 위한 짧은 지연
                    const timeout = setTimeout(() => {
                        if (activeIndex === targetIndex && !currentPane) {
                            isTransitioning = false;
                        }
                        transitionTimeouts.delete(timeout);
                    }, 100);
                    transitionTimeouts.add(timeout);
                }
            });
        };

        tabItems.forEach((item, index) => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                switchTab(index);
            });
        });
    }
};

const MobileMenu = {
    init() {
        const toggleButton = document.querySelector('.menu-toggle');
        const overlay = document.getElementById('mobileMenu');
        if (!toggleButton || !overlay) return;

        const closeButtons = overlay.querySelectorAll('.mobile-menu-close, .mobile-menu-secondary');
        const firstFocusable = overlay.querySelector('a, button');
        let previousActiveElement = null;

        const openMenu = () => {
            overlay.classList.add('active');
            overlay.setAttribute('aria-hidden', 'false');
            toggleButton.setAttribute('aria-expanded', 'true');
            previousActiveElement = document.activeElement;
            setTimeout(() => {
                firstFocusable?.focus({ preventScroll: true });
            }, 10);
            document.body.style.overflow = 'hidden';
        };

        const closeMenu = () => {
            overlay.classList.remove('active');
            overlay.setAttribute('aria-hidden', 'true');
            toggleButton.setAttribute('aria-expanded', 'false');
            toggleButton.focus({ preventScroll: true });
            document.body.style.overflow = '';
        };

        toggleButton.addEventListener('click', () => {
            const isExpanded = toggleButton.getAttribute('aria-expanded') === 'true';
            if (isExpanded) {
                closeMenu();
            } else {
                openMenu();
            }
        });

        closeButtons.forEach(button => {
            button.addEventListener('click', () => closeMenu());
        });

        overlay.addEventListener('click', (event) => {
            if (event.target === overlay) {
                closeMenu();
            }
        });

        const navLinks = overlay.querySelectorAll('.mobile-menu-nav a[href^="#"], .mobile-menu-quickactions a[href^="#"]');
        navLinks.forEach(link => {
            link.addEventListener('click', () => closeMenu());
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && overlay.classList.contains('active')) {
                closeMenu();
            }
        });

        window.addEventListener('resize', () => {
            if (window.innerWidth > 800 && overlay.classList.contains('active')) {
                closeMenu();
            }
        });
    }
};

const DemoFloat = {
    init() {
        const demoBtn = document.getElementById('demo-btn');
        const demoFloat = document.getElementById('demoFloat');
        const resultSection = document.getElementById('resultSection');
        if (!demoFloat) return;

        const closeBtn = demoFloat.querySelector('.demo-float-close');
        const input = demoFloat.querySelector('.badge-input');
        let manualDismissed = false;

        const showFloat = () => {
            if (manualDismissed || demoFloat.classList.contains('active')) return;
            demoFloat.classList.add('active');
            demoFloat.setAttribute('aria-hidden', 'false');
            setTimeout(() => input?.focus({ preventScroll: true }), 120);
        };

        const hideFloat = () => {
            if (!demoFloat.classList.contains('active')) return;
            demoFloat.classList.remove('active');
            demoFloat.setAttribute('aria-hidden', 'true');
        };

        if (demoBtn) {
            demoBtn.addEventListener('click', (event) => {
                event.preventDefault();
                manualDismissed = false;
                if (resultSection) {
                    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                } else {
                    showFloat();
                }
            });
        }

        closeBtn?.addEventListener('click', () => {
            manualDismissed = true;
            hideFloat();
        });

        if (resultSection) {
            const sectionObserver = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        if (!manualDismissed) showFloat();
                    } else {
                        hideFloat();
                        manualDismissed = false;
                    }
                });
            }, { threshold: 0.35 });
            sectionObserver.observe(resultSection);
        }
    }
};

// FAST 모드/백그라운드 처리 제거됨 – 모든 분석은 단일 정밀 파이프라인으로 동작합니다.

function mergeObjects(target, source) {
    const base = (target && typeof target === 'object' && !Array.isArray(target)) ? { ...target } : {};
    if (!source || typeof source !== 'object') {
        return base;
    }
    Object.keys(source).forEach(key => {
        const value = source[key];
        if (Array.isArray(value)) {
            base[key] = value.map(item => (item && typeof item === 'object') ? { ...item } : item);
        } else if (value && typeof value === 'object') {
            base[key] = mergeObjects(base[key], value);
        } else if (value !== undefined) {
            base[key] = value;
        }
    });
    return base;
}

function alignResultData(raw) {
    if (!raw || typeof raw !== 'object') return raw;
    const data = { ...raw };
    const bundle = raw.bundle && typeof raw.bundle === 'object' ? raw.bundle : null;
    if (!bundle) {
        console.warn("[alignResultData] Truth bundle is missing");
        // 번들이 없어도 results가 있으면 최소한의 표시는 가능하도록 함
        return data;
    }

    const aliasMap = (raw.raw_json && raw.raw_json.sub_label_map) || {};

    if (!data.bundle || data.bundle !== bundle) {
        data.bundle = mergeObjects({}, bundle);
    }
    
    // [GENIUS FIX] Truth Data Preservation
    // Expert View와 Business View 간의 데이터 불일치를 해결하기 위해
    // 서버에서 받은 원본 bundle 데이터를 truth 필드에 그대로 보존합니다.
    data.truth = {
        main_dist: bundle.main_dist || null,
        sub_top: Array.isArray(bundle.sub_top) ? bundle.sub_top : null,
        // [CRITICAL] 구조화된 주석 데이터 우선 사용 (서버에서 포맷팅됨)
        sentence_annotations_structured: Array.isArray(bundle.sentence_annotations_structured) 
            ? bundle.sentence_annotations_structured 
            : null,
        transitions_structured: Array.isArray(bundle.transitions_structured) 
            ? bundle.transitions_structured 
            : null,
        why_lines: Array.isArray(bundle.why_lines) ? bundle.why_lines : null,
        reasoning_path_lines: Array.isArray(bundle.reasoning_path_lines) 
            ? bundle.reasoning_path_lines 
            : null,
        sub_top10_lines: Array.isArray(bundle.sub_top10_lines) 
            ? bundle.sub_top10_lines 
            : null,
        flow_ssot: bundle.flow_ssot || null,
        triggers: bundle.triggers || null,
        products: bundle.products || null,
        reports: bundle.reports || null,
        meta: bundle.meta || null,
    };

    // [FIX] Web View Data Synchronization
    // 웹 뷰(Business View)에서 사용하는 필드들도 Truth 기반으로 재구성하여 일관성 보장

    // 1. Sentence Annotations (문장 주석)
    // bundle.sentence_annotations_structured가 가장 정확한(서버에서 처리된) 데이터임.
    // 이를 data.sentence_annotations_structured로 승격시켜 렌더러가 사용하게 함.
    if (data.truth.sentence_annotations_structured) {
        data.sentence_annotations_structured = data.truth.sentence_annotations_structured;
        
        // [GENIUS FIX] Recursive Safety Check
        // 만약 승격된 데이터 안에도 'sub_'가 숨어있다면 여기서 미리 발본색원합니다.
        data.sentence_annotations_structured.forEach(item => {
            if (item.sub_label && item.sub_label.includes('sub_')) {
                 item.sub_label = normalizeSubLabel(item.main, item.sub_label);
            }
            if (item.sub && item.sub.includes('sub_')) {
                 item.sub = normalizeSubLabel(item.main, item.sub);
            }
        });
    }

    // 2. Transitions (감정 전이)
    // bundle.transitions_structured가 있으면 우선 사용
    if (data.truth.transitions_structured) {
        data.transitions_structured = data.truth.transitions_structured;
        
        // [GENIUS FIX] Transition Safety Check
        // 전이 데이터 내부의 'sub_'도 발본색원
        data.transitions_structured.forEach(t => {
            if (t.from_sub && t.from_sub.includes('sub_')) {
                t.from_sub = normalizeSubLabel(t.from_main, t.from_sub);
            }
            if (t.to_sub && t.to_sub.includes('sub_')) {
                t.to_sub = normalizeSubLabel(t.to_main, t.to_sub);
            }
        });
    }

    // ★★★ bundle.main_dist 최우선 사용 (test.py 원본 데이터 보존) ★★★
    if (bundle.main_dist && typeof bundle.main_dist === 'object') {
        const mainDistribution = {};
        let total = 0;
        Object.entries(bundle.main_dist).forEach(([key, value]) => {
            const label = mapMainLabel(key);
            const score = Number(value);
            if (!label || Number.isNaN(score) || score <= 0) return;
            // 같은 라벨이 여러 번 나타날 수 있으므로 합산
            mainDistribution[label] = (mainDistribution[label] || 0) + score;
            total += score;
        });
        if (Object.keys(mainDistribution).length) {
            // 합이 1.0에 가까우면 이미 정규화된 것으로 간주 (그대로 사용)
            // 합이 1.0보다 크면 정규화 필요 (퍼센트 값이거나 합산된 값)
            if (Math.abs(total - 1.0) < 0.01) {
                // 이미 정규화됨 (0-1 사이) → 그대로 사용
                data.main_distribution = mainDistribution;
            } else if (total > 1.0) {
                // 정규화 필요
                const normalized = {};
                Object.entries(mainDistribution).forEach(([k, v]) => {
                    normalized[k] = v / total;
                });
                data.main_distribution = normalized;
            } else {
                // 합이 1.0보다 작으면 그대로 사용 (일부 감정만 있는 경우)
                data.main_distribution = mainDistribution;
            }
            
            const poster = mergeObjects(data.poster, {});
            poster.main_distribution = data.main_distribution;
            const sortedMain = Object.keys(data.main_distribution).sort((a, b) => (data.main_distribution[b] || 0) - (data.main_distribution[a] || 0));
            if (sortedMain.length) {
                poster.main = sortedMain[0];
            }
            data.poster = poster;
        }
    }

    // [GENIUS FIX] Sub-distribution Truth Preservation
    // sub_top 데이터를 처리할 때 aliasMap에 의존하지 않고 원본 라벨을 보존합니다.
    // 이미 서버의 _format_sub_label에서 사람이 읽기 쉬운 형태로 변환되었으므로
    // 클라이언트 측에서 불필요한 매핑을 수행하면 오히려 정보가 손실될 수 있습니다.
    if (Array.isArray(bundle.sub_top) && bundle.sub_top.length) {
        const subDistribution = {};
        bundle.sub_top.forEach(entry => {
            // entry.sub가 이미 포맷된 라벨(예: "기쁨", "비통함")일 가능성이 높음
            const rawSub = entry.sub || entry.name || entry.label;
            const score = Number(entry.p ?? entry.score ?? entry.value);
            
            if (!rawSub || Number.isNaN(score)) return;
            
            // aliasMap 체크는 하되, 없으면 rawSub 그대로 사용 (test.py 결과 신뢰)
            const normalized = (aliasMap && aliasMap[rawSub]) ? aliasMap[rawSub] : rawSub;
            
            // "sub_" 형식이 남아있으면 기본값 매핑 시도 (최후의 보루)
            if (typeof normalized === 'string' && normalized.includes('sub_')) {
                 // 이 경우는 이미 서버에서 처리되었어야 함. 로그만 남김.
                 // [GENIUS FIX] Client-side Fallback
                 // 만약 서버에서 변환이 실패했다면, 클라이언트 측 매핑을 시도합니다.
                 // aliasMap[rawSub]가 없다면 DEFAULT_SUB_LABEL에서 찾습니다.
                 // entry에 main 정보가 없으므로 추론해야 할 수 있습니다.
                 console.warn('[alignResultData] Unformatted sub-label detected:', normalized);
                 
                 // 긴급 복구 로직 추가
                 const possibleMain = Object.keys(DEFAULT_SUB_LABEL).find(k => normalized.startsWith(k)) || '희';
                 normalized = DEFAULT_SUB_LABEL[possibleMain];
            }
            
            subDistribution[normalized] = score;
        });
        if (Object.keys(subDistribution).length) {
            data.sub_distribution = subDistribution;
        }
    }

    if (bundle.products && typeof bundle.products === 'object') {
        data.products = mergeObjects(data.products, bundle.products);
    }

    if (bundle.triggers && typeof bundle.triggers === 'object') {
        data.triggers = mergeObjects(data.triggers, bundle.triggers);
    }

    if (bundle.weight_drivers && typeof bundle.weight_drivers === 'object') {
        data.weight_drivers = mergeObjects(data.weight_drivers, bundle.weight_drivers);
    }

    // raw_json.results를 data.results에 병합 (test.py 결과와 동일하게)
    const rawJson = raw.raw_json || raw;
    if (rawJson.results && typeof rawJson.results === 'object') {
        data.results = mergeObjects(data.results || {}, rawJson.results);
    }

    return data;
}

// AnalysisController
const AnalysisController = {
    state: {},

    init() {
        const form = document.getElementById('analysisForm');
        const input = document.getElementById('analysisInput');
        const reportBtn = document.querySelector('.report-btn');
        const notificationBtn = document.querySelector('.notification-btn');
        const modal = document.getElementById('analysisModal');
        const headerLogo = document.querySelector('.logo-link');
        const mobileLogo = document.querySelector('.mobile-menu-logo');
        
        if (!form || !input || !modal) {
            console.warn('[AnalysisController] Required elements not found');
            return;
        }

        const submitBtn = form.querySelector('.badge-btn');
        const closeBtn = modal.querySelector('.analysis-modal__close');
        const badge = document.querySelector('.badge');
        const progress = document.getElementById('analysisProgress');
        const progressText = document.getElementById('analysisProgressText');
        const progressSub = document.getElementById('analysisProgressSub');
        const progressIcon = document.getElementById('analysisProgressIcon');
        const notificationBadge = notificationBtn ? notificationBtn.querySelector('.notification-badge') : null;

        if (!badge) {
            console.warn('[AnalysisController] Badge element not found');
            return;
        }

        form.setAttribute('action', 'javascript:void(0);');
        form.setAttribute('novalidate', 'novalidate');

        // 진행 상태 관련 요소
        const progressBarContainer = document.getElementById('analysisProgressBarContainer');
        const progressBarFill = document.getElementById('analysisProgressBarFill');
        const progressSteps = document.getElementById('analysisProgressSteps');
        const progressElapsed = document.getElementById('analysisProgressElapsed');
        const progressEstimated = document.getElementById('analysisProgressEstimated');

        const cancelBtn = document.getElementById('analysisCancelBtn');
        
        this.state = {
            maxRetries: 2, // 최대 재시도 횟수
            form,
            input,
            submitBtn,
            reportBtn,
            notificationBtn,
            notificationBadge,
            headerLogo,
            mobileLogo,
            mobileLogoDefaultText: mobileLogo ? mobileLogo.textContent : '',
            modal,
            closeBtn,
            badge,
            progress,
            progressText,
            progressSub,
            progressIcon,
            progressBarContainer,
            progressBarFill,
            progressSteps,
            progressElapsed,
            progressEstimated,
            cancelBtn,
            progressPersistent: false,
            progressTimeout: null,
            currentMode: 'balanced',
            lastModalOpenAt: 0,
            analysisStartTime: null,
            progressTimer: null,
            currentStep: 0,
            abortController: null, // 분석 취소를 위한 AbortController
            isAnalyzing: false, // 분석 중 상태 플래그
            currentJobId: null, // 현재 실행 중인 작업 ID (서버 취소용)
            isCancelled: false // 취소 플래그 (재시도 방지용)
        };

        // 폼 제출 이벤트
        form.addEventListener('submit', (event) => {
            event.preventDefault();
            if (typeof event.stopPropagation === 'function') event.stopPropagation();
            if (typeof event.stopImmediatePropagation === 'function') event.stopImmediatePropagation();
            const text = input.value.trim();
            if (!text) {
                input.focus();
                return;
            }
            if (text.length > 300) {
                alert('텍스트는 최대 300자까지 입력할 수 있습니다. 내용을 조금만 줄여주세요.');
                input.focus();
                return;
            }
            this.runAnalysis(text);
        });

        if (input) {
            input.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    if (typeof form.requestSubmit === 'function') {
                        form.requestSubmit();
                    } else {
                        form.dispatchEvent(new Event('submit', { cancelable: true }));
                    }
                }
            });
        }

        // 리포트 버튼 클릭 시 마지막 결과 표시
        if (reportBtn) {
            reportBtn.addEventListener('click', () => {
                if (this.lastResult) {
                    if (AnalysisRenderer) {
                        AnalysisRenderer.render(this.lastResult);
                        const modeLabel = (this.lastResult.meta && this.lastResult.meta.mode)
                            ? String(this.lastResult.meta.mode).toUpperCase()
                            : 'BALANCED';
                        AnalysisRenderer.setStatus(`${modeLabel} 모드 분석 결과를 불러왔습니다.`);
                    }
                    this.openModal();
                    this.clearReportAlert();
                } else {
                    alert('먼저 텍스트를 입력하고 분석을 실행해주세요.');
                }
            });
        }

        if (notificationBtn) {
            notificationBtn.addEventListener('click', () => {
                if (this.lastResult) {
                    if (AnalysisRenderer) {
                        AnalysisRenderer.render(this.lastResult);
                        const modeLabel = (this.lastResult.meta && this.lastResult.meta.mode)
                            ? String(this.lastResult.meta.mode).toUpperCase()
                            : 'BALANCED';
                        AnalysisRenderer.setStatus(`${modeLabel} 모드 분석 결과를 불러왔습니다.`);
                    }
                    this.openModal();
                    this.clearReportAlert();
                } else {
                    alert('먼저 텍스트를 입력하고 분석을 실행해주세요.');
                }
            });
        }

        input.addEventListener('input', () => {
            const hasValue = input.value.trim().length > 0;
            badge.classList.toggle('is-loaded', hasValue);
        });
        badge.classList.toggle('is-loaded', input.value.trim().length > 0);
        this.clearReportAlert();

        // 닫기 버튼
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closeModal());
        }

        // 오버레이 클릭으로 닫기
        const overlay = modal.querySelector('.analysis-modal__overlay');
        if (overlay) {
            overlay.addEventListener('click', () => {
                const now = Date.now();
                if (now - (this.state.lastModalOpenAt || 0) < 400) {
                    return;
                }
                this.closeModal();
            });
        }

        // ESC 키로 닫기
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && modal.classList.contains('active')) {
                this.closeModal();
            }
        });

        // 취소 버튼 클릭 이벤트
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => {
                this.cancelAnalysis();
            });
        }

        // 페이지 이탈 시 분석 중단 처리
        window.addEventListener('beforeunload', (event) => {
            if (this.state.isAnalyzing) {
                // 분석 중이면 서버에 취소 요청 전송
                const jobId = this.state.currentJobId;
                if (jobId) {
                    // fetch with keepalive를 사용하여 페이지 이탈 시에도 요청 전송 보장
                    try {
                        const cancelEndpoint = resolveApiUrl(`/api/job/${jobId}/cancel`);
                        // keepalive 옵션으로 페이지 이탈 후에도 요청이 완료되도록 보장
                        fetch(cancelEndpoint, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ reason: '페이지 이탈' }),
                            keepalive: true // 페이지 이탈 후에도 요청 완료 보장
                        }).catch(error => {
                            console.error('[AnalysisController] 페이지 이탈 시 취소 요청 실패:', error);
                        });
                        console.log('[AnalysisController] 페이지 이탈 시 서버 취소 요청 전송:', jobId);
                    } catch (error) {
                        console.error('[AnalysisController] 페이지 이탈 시 취소 요청 실패:', error);
                    }
                }
                
                // 클라이언트 측 취소 처리
                if (this.state.abortController) {
                    this.state.abortController.abort();
                }
                // 새로고침 허용 (확인 다이얼로그 표시하지 않음)
                return;
            }
        });

        // 미니 프로그레스 설정 (스크롤 옵저버 등)
        this.setupMiniProgress();
    },

    setupMiniProgress() {
        const btn = document.querySelector('.notification-btn');
        if (!btn) return;
        
        // SVG 링 주입 (없을 경우)
        if (!btn.querySelector('.mini-progress-container')) {
            const svgHtml = `
                <div class="mini-progress-container">
                    <svg class="mini-progress-ring" width="34" height="34" viewBox="0 0 32 32">
                        <circle class="mini-progress-ring__bg" stroke="rgba(255,255,255,0.1)" stroke-width="2" fill="transparent" r="14" cx="16" cy="16"/>
                        <circle class="mini-progress-ring__circle" stroke="#10b981" stroke-width="2" fill="transparent" r="14" cx="16" cy="16" stroke-dasharray="87.96" stroke-dashoffset="87.96"/>
                    </svg>
                </div>
            `;
            btn.insertAdjacentHTML('beforeend', svgHtml);
        }
        
        // Intersection Observer 설정
        const hero = document.querySelector('.main-hero');
        if (!hero) return;

        // 기존 옵저버가 있다면 해제
        if (this.observer) {
            this.observer.disconnect();
        }

        this.observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                const notificationBtn = document.querySelector('.notification-btn');
                if (!notificationBtn) return;

                if (this.state.isAnalyzing) {
                    if (!entry.isIntersecting) {
                        // Hero 영역을 벗어남 -> 미니 프로그레스 활성화
                        notificationBtn.classList.add('mini-active');
                    } else {
                        // Hero 영역으로 돌아옴 -> 미니 프로그레스 비활성화
                        notificationBtn.classList.remove('mini-active');
                    }
                } else {
                    // 분석 중이 아님 -> 비활성화
                    notificationBtn.classList.remove('mini-active');
                }
            });
        }, { threshold: 0.1 }); // 10% 정도 보일 때 트리거
        
        this.observer.observe(hero);
    },

    updateMiniProgress(percent) {
        const circle = document.querySelector('.mini-progress-ring__circle');
        if (!circle) return;
        
        const radius = 14;
        const circumference = 2 * Math.PI * radius; // ~87.96
        const offset = circumference - (percent / 100) * circumference;
        
        circle.style.strokeDashoffset = offset;
    },

    setAnalyzing(flag) {
        const { input, submitBtn, reportBtn, notificationBtn, notificationBadge, cancelBtn, headerLogo, mobileLogo, mobileLogoDefaultText } = this.state;
        this.state.isAnalyzing = flag;
        
        if (input) {
            input.disabled = flag;
            input.setAttribute('aria-disabled', flag ? 'true' : 'false');
        }

        // 미니 프로그레스 해제
        if (!flag && notificationBtn) {
            notificationBtn.classList.remove('mini-active');
            if (flag) {
                // 분석 시작 시 completed 상태 제거
                notificationBtn.classList.remove('completed');
            }
        }
        if (submitBtn) {
            submitBtn.disabled = flag;
            submitBtn.setAttribute('aria-disabled', flag ? 'true' : 'false');
        }
        if (notificationBtn) {
            notificationBtn.disabled = flag;
            notificationBtn.setAttribute('aria-disabled', flag ? 'true' : 'false');
        }
        if (notificationBadge && flag) {
            notificationBadge.classList.remove('active');
        }
        if (flag) {
            this.clearReportAlert();
        }
        
        // 취소 버튼 표시/숨김
        if (cancelBtn) {
            if (flag) {
                cancelBtn.style.display = 'flex';
            } else {
                cancelBtn.style.display = 'none';
            }
        }
        
        // 리포트 버튼 상태 업데이트
        // 분석 시작 시에는 analyzing 클래스를 추가하지 않음 (완료 시점에만 completed 추가)
        if (reportBtn) {
            if (flag) {
                // 분석 중에는 analyzing 클래스를 추가하지 않음
                reportBtn.classList.remove('completed');
            } else {
                reportBtn.classList.remove('analyzing');
            }
        }
        if (notificationBtn && flag) {
            notificationBtn.classList.remove('completed');
        }

        // 분석 상태에 따른 알림 버튼 접근성 라벨/툴팁 업데이트
        if (notificationBtn) {
            if (flag) {
                notificationBtn.setAttribute('aria-label', '분석중입니다. (분석 완료 후 리포트를 확인할 수 있습니다)');
                notificationBtn.setAttribute('title', '분석중입니다.');
            } else {
                notificationBtn.setAttribute('aria-label', '분석 리포트 보기');
                notificationBtn.removeAttribute('title');
            }
        }

        // 헤더 로고 / 모바일 메뉴 로고에 분석 상태 표시
        if (headerLogo) {
            if (flag) {
                headerLogo.classList.add('analyzing');
            } else {
                headerLogo.classList.remove('analyzing');
            }
        }

        if (mobileLogo) {
            if (flag) {
                mobileLogo.classList.add('analyzing');
                mobileLogo.textContent = '분석중입니다';
            } else {
                mobileLogo.classList.remove('analyzing');
                mobileLogo.textContent = mobileLogoDefaultText || 'AI emotion standards authority';
            }
        }
    },

    async cancelAnalysis(silent = false) {
        if (!this.state.isAnalyzing || this.state.isCancelled) {
            return;
        }

        // 취소 플래그 설정 (재시도 방지)
        this.state.isCancelled = true;

        const jobId = this.state.currentJobId;

        // 1. 클라이언트 측 요청 취소 (AbortController)
        if (this.state.abortController) {
            this.state.abortController.abort();
            this.state.abortController = null;
        }

        // 2. 서버 측 작업 취소 (job_id가 있으면)
        if (jobId) {
            try {
                const cancelEndpoint = resolveApiUrl(`/api/job/${jobId}/cancel`);
                const cancelResponse = await fetch(cancelEndpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ reason: '사용자 요청' }),
                    signal: new AbortController().signal // 취소 요청 자체는 취소되지 않도록
                });
                
                if (cancelResponse.ok) {
                    const cancelData = await cancelResponse.json();
                    console.log('[AnalysisController] 서버 작업 취소 성공:', cancelData);
                } else {
                    console.warn('[AnalysisController] 서버 작업 취소 실패:', cancelResponse.status);
                }
            } catch (error) {
                console.error('[AnalysisController] 서버 작업 취소 요청 실패:', error);
                // 서버 취소 실패해도 클라이언트 측 취소는 진행
            }
        }

        // 상태 초기화
        this.setAnalyzing(false);
        this.stopProgressTimer();
        // 프로그레스바 강제 정리
        this.state.progressPersistent = false;
        this.clearProgress(true);
        this.state.currentJobId = null;
        
        // 프로그레스 UI 강제 숨김
        const { progress } = this.state;
        if (progress) {
            progress.classList.remove('active');
            progress.setAttribute('aria-hidden', 'true');
        }

        // [FIX] 모달(프로그레스 박스) 즉시 닫기
        this.closeModal();

        if (!silent) {
            this.setStatus('분석이 취소되었습니다.', false);
            // 취소 메시지 표시 후 자동 클리어 스케줄링 대신, 즉시 상태 업데이트만 수행
            // this.showProgress(...) 호출 제거하여 모달이 다시 뜨지 않도록 함
        }

        console.log('[AnalysisController] 분석 취소됨');
    },

    openModal() {
        const { modal } = this.state;
        if (!modal) return;
        modal.classList.add('active');
        modal.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        this.state.lastModalOpenAt = Date.now();
    },

    closeModal() {
        const { modal, input } = this.state;
        if (!modal) return;
        modal.classList.remove('active');
        modal.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        if (input) input.focus();
    },

    async runAnalysis(text, retryCount = 0) {
        // 취소 플래그 확인 (재시도 방지)
        if (this.state.isCancelled && retryCount === 0) {
            // 새로운 분석 시작 시 취소 플래그 초기화
            this.state.isCancelled = false;
        } else if (this.state.isCancelled) {
            // 재시도 중 취소 플래그가 설정되어 있으면 재시도 중단
            console.log('[AnalysisController] 취소됨 - 재시도 중단');
            return;
        }

        if (this.state?.modal?.classList.contains('active')) {
            this.closeModal();
        }

        const endpoint = resolveApiUrl('/api/analyze');
        const selectedMode = 'balanced';
        const modeInput = document.getElementById('analysisMode');
        if (modeInput) {
            modeInput.value = selectedMode;
        }

        this.state.currentMode = selectedMode;
        
        // [Client-Side Job ID] 클라이언트에서 UUID 생성하여 서버와 공유
        // 이를 통해 요청 직후(응답 전)에도 취소 가능하도록 함
        const clientJobId = crypto.randomUUID ? crypto.randomUUID() : `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.state.currentJobId = clientJobId;
        
        // 재시도 중이 아닐 때만 초기 진행 상태 표시
        if (retryCount === 0) {
            this.showProgress({
                mode: selectedMode,
                primary: '정밀 분석을 준비하고 있습니다.',
                secondary: '모든 모듈이 순차적으로 실행됩니다.',
                persistent: true,
                showDetailed: true,
                stepIndex: null
            });
        } else {
            this.showProgress({
                mode: selectedMode,
                primary: `분석 재시도 중... (${retryCount}/${this.state.maxRetries || 2})`,
                secondary: '서버 연결을 다시 시도하고 있습니다.',
                persistent: true,
                showDetailed: false
            });
        }

        this.setAnalyzing(true);
        this.setStatus(retryCount === 0 ? 'BALANCED 정밀 분석을 요청했습니다.' : `재시도 ${retryCount}/${this.state.maxRetries || 2}`);

        console.log('[AnalysisController] 분석 요청:', endpoint, '| mode =', selectedMode, '| job_id =', clientJobId, '| retry =', retryCount);

        // DEMO API Key 처리: localStorage에 저장된 키를 우선 사용하고, 없으면 한 번만 입력받습니다.
        let demoKey = null;
        try {
            if (typeof window !== 'undefined' && window.localStorage) {
                demoKey = window.localStorage.getItem('demoApiKey');
            }
        } catch (e) {
            demoKey = null;
        }

        if (!demoKey) {
            // 이력서/포트폴리오에 기재된 데모 키를 입력하도록 안내
            demoKey = window.prompt('이 데모는 접근 키가 필요합니다. 이력서나 안내에 표시된 데모 키를 입력해주세요.');
            if (!demoKey) {
                this.setAnalyzing(false);
                this.showError('데모 키가 입력되지 않아 분석을 진행할 수 없습니다.');
                return;
            }
            try {
                if (typeof window !== 'undefined' && window.localStorage) {
                    window.localStorage.setItem('demoApiKey', demoKey);
                }
            } catch (e) {
                // localStorage 실패는 치명적이지 않으므로 무시
            }
        }

        try {
            // 타임아웃 설정 (긴 텍스트 분석을 위해 충분한 시간 제공)
            // BALANCED 모드는 heavy pipeline을 사용하므로 최대 10분까지 허용
            const controller = new AbortController();
            this.state.abortController = controller; // state에 저장하여 취소 가능하게 함
            const timeoutMs = selectedMode === 'balanced' ? 600000 : 300000; // balanced: 10분, fast: 5분
            const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

            let response;
            try {
                response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Job-ID': clientJobId, // [Client-Side Job ID] 헤더로 전달
                        'X-API-KEY': demoKey
                    },
                    body: JSON.stringify({ text, mode: selectedMode }),
                    signal: controller.signal
                });
            } catch (fetchError) {
                clearTimeout(timeoutId);
                this.state.abortController = null; // 정리
                
                // 취소 플래그 확인 (재시도 방지)
                if (this.state.isCancelled) {
                    console.log('[AnalysisController] 취소됨 - 요청 중단');
                    return;
                }
                
                // 사용자가 취소한 경우
                if (fetchError.name === 'AbortError' && controller.signal.aborted) {
                    // 취소 버튼을 눌렀거나 페이지 이탈로 인한 취소
                    if (!this.state.isAnalyzing || this.state.isCancelled) {
                        // 이미 cancelAnalysis에서 처리됨
                        console.log('[AnalysisController] 분석 취소됨 (AbortError)');
                        return;
                    }
                    // 취소 플래그가 설정되지 않았지만 AbortError가 발생한 경우
                    // (예: 타임아웃 등) - 사용자에게 알림
                    console.log('[AnalysisController] 요청 중단됨 (AbortError) - 타임아웃 또는 네트워크 오류');
                    this.showError('분석 시간이 초과되었습니다. 서버에서는 분석이 완료되었을 수 있으니 잠시 후 다시 시도해주세요.');
                    this.setAnalyzing(false);
                    return;
                }
                
                // ERR_CONNECTION_RESET 또는 네트워크 오류 처리
                if (fetchError.name === 'AbortError') {
                    throw new Error('요청 시간 초과: 분석에 너무 오래 걸리고 있습니다. 텍스트 길이를 줄이거나 잠시 후 다시 시도해주세요.');
                } else if (fetchError.message && (
                    fetchError.message.includes('Failed to fetch') ||
                    fetchError.message.includes('ERR_CONNECTION_RESET') ||
                    fetchError.message.includes('NetworkError') ||
                    fetchError.message.includes('network')
                )) {
                    // 네트워크 오류는 재시도 가능 (취소 플래그 확인)
                    if (this.state.isCancelled) {
                        console.log('[AnalysisController] 취소됨 - 재시도 중단');
                        return;
                    }
                    const maxRetries = this.state.maxRetries || 2;
                    if (retryCount < maxRetries) {
                        console.warn(`[AnalysisController] 네트워크 오류 발생, ${1000 * (retryCount + 1)}ms 후 재시도...`);
                        await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1))); // 지수 백오프
                        // 재시도 전 취소 플래그 재확인
                        if (this.state.isCancelled) {
                            console.log('[AnalysisController] 취소됨 - 재시도 중단');
                            return;
                        }
                        return this.runAnalysis(text, retryCount + 1);
                    }
                    throw new Error('서버 연결에 실패했습니다. 서버가 실행 중인지 확인해주세요.');
                }
                throw fetchError;
            }
            
            clearTimeout(timeoutId);
            this.state.abortController = null; // 정리

            // 응답 본문 파싱 시도
            let responseData;
            try {
                const responseText = await response.text();
                if (!responseText) {
                    throw new Error('서버로부터 빈 응답을 받았습니다.');
                }
                
                try {
                    responseData = JSON.parse(responseText);
                } catch (parseError) {
                    console.error('[AnalysisController] JSON 파싱 실패:', responseText.substring(0, 200));
                    throw new Error(`서버 응답 형식 오류: ${responseText.substring(0, 100)}`);
                }
            } catch (parseError) {
                if (parseError.message.includes('서버')) {
                    throw parseError;
                }
                throw new Error(`응답 처리 실패: ${parseError.message}`);
            }

            // HTTP 상태 코드 확인
            if (!response.ok) {
                // 취소 플래그 확인
                if (this.state.isCancelled) {
                    console.log('[AnalysisController] 취소됨 - 응답 처리 중단');
                    return;
                }
                
                const errorMsg = responseData?.error || responseData?.detail || `HTTP ${response.status}`;
                console.error('[AnalysisController] 응답 오류:', response.status, errorMsg);
                
                // 취소된 경우 (499 상태 코드)
                if (response.status === 499 || responseData?.cancelled) {
                    console.log('[AnalysisController] 서버에서 작업 취소 확인됨');
                    return;
                }

                // 401: 데모 키 인증 실패 - 재입력 기회 제공
                if (response.status === 401) {
                    try {
                        if (typeof window !== 'undefined' && window.localStorage) {
                            window.localStorage.removeItem('demoApiKey');
                        }
                    } catch (e) {
                        // ignore
                    }
                    
                    // 최대 3회까지 재시도 허용
                    const authRetryCount = retryCount || 0;
                    if (authRetryCount < 3) {
                        const newKey = window.prompt(
                            `비밀번호가 틀렸습니다. (${authRetryCount + 1}/3)\n\n이력서에 기재된 올바른 비밀번호를 다시 입력해주세요.`
                        );
                        if (newKey) {
                            try {
                                if (typeof window !== 'undefined' && window.localStorage) {
                                    window.localStorage.setItem('demoApiKey', newKey);
                                }
                            } catch (e) {
                                // ignore
                            }
                            // 재시도 (authRetryCount 증가)
                            return this.runAnalysis(text, authRetryCount + 1);
                        }
                    }
                    
                    this.setAnalyzing(false);
                    this.showError('비밀번호 인증에 실패했습니다. 이력서에 기재된 올바른 비밀번호를 확인해주세요.');
                    return;
                }
                
                // 5xx 오류는 재시도 가능 (취소 플래그 확인)
                if (response.status >= 500 && response.status < 600) {
                    if (this.state.isCancelled) {
                        console.log('[AnalysisController] 취소됨 - 재시도 중단');
                        return;
                    }
                    const maxRetries = this.state.maxRetries || 2;
                    if (retryCount < maxRetries) {
                        console.warn(`[AnalysisController] 서버 오류 발생 (${response.status}), ${2000 * (retryCount + 1)}ms 후 재시도...`);
                        await new Promise(resolve => setTimeout(resolve, 2000 * (retryCount + 1)));
                        // 재시도 전 취소 플래그 재확인
                        if (this.state.isCancelled) {
                            console.log('[AnalysisController] 취소됨 - 재시도 중단');
                            return;
                        }
                        return this.runAnalysis(text, retryCount + 1);
                    }
                }
                
                throw new Error(`서버 오류 (${response.status}): ${errorMsg}`);
            }

            // success 필드 확인
            if (responseData && responseData.success === false) {
                // 취소된 경우 특별 처리
                if (responseData.cancelled) {
                    console.log('[AnalysisController] 서버에서 작업 취소 확인됨');
                    // 이미 취소 처리되었으므로 추가 작업 없음
                    return;
                }
                const errorMsg = responseData.error || responseData.master_report || '분석 실패';
                throw new Error(errorMsg);
            }

            // 작업 ID 저장 (취소 가능하도록)
            if (responseData.job_id) {
                this.state.currentJobId = responseData.job_id;
                console.log('[AnalysisController] 작업 ID 저장:', responseData.job_id);
            }

            // 데이터 정렬 및 처리
            let data = alignResultData(responseData);
            console.log(
                '[AnalysisController] 응답 meta:',
                data && data.meta ? JSON.parse(JSON.stringify(data.meta)) : null,
                '| refined =',
                data && data.meta ? data.meta.layered_refinement : undefined
            );

            data.text = text;
            data.inputText = text;

            // 실제 완료된 모듈을 기반으로 프로그레스 업데이트
            if (data.module_details && Array.isArray(data.module_details)) {
                this.updateProgressFromModuleDetails(data.module_details, selectedMode);
            }

            this.lastResult = data;
            this.openModal();

            const { reportBtn } = this.state;
            if (reportBtn) {
                reportBtn.classList.remove('analyzing');
                reportBtn.classList.add('completed');
            }
            const { notificationBtn } = this.state;
            if (notificationBtn) {
                notificationBtn.classList.add('completed');
            }

            setTimeout(() => {
                const content = document.querySelector('.analysis-modal__body');
                if (content) {
                    content.scrollTop = 0;
                }
            }, 100);

            if (AnalysisRenderer) {
                AnalysisRenderer.render(data);
                AnalysisRenderer.setStatus('BALANCED 모드 정밀 분석 결과입니다.');
            }

            if (this.state.progressBarFill) {
                this.state.progressBarFill.style.width = '100%';
            }
            this.stopProgressTimer();
            this.showProgress({
                mode: 'balanced',
                primary: '정밀 분석이 완료되었습니다.',
                secondary: '결과가 최신 상태로 반영되었습니다.',
                persistent: false,
                showDetailed: false
            });
            this.scheduleProgressClear();
            this.clearReportAlert();
        } catch (error) {
            const { reportBtn } = this.state;

            // 취소 플래그 확인 (재시도 방지 및 오류 로그 방지)
            if (this.state.isCancelled) {
                console.log('[AnalysisController] 취소됨 - 오류 처리 중단');
                if (reportBtn) {
                    reportBtn.classList.remove('analyzing', 'completed');
                }
                return;
            }

            // 취소 관련 오류는 오류로 처리하지 않음
            if (error.message && (
                error.message.includes('취소') ||
                error.message.includes('cancelled') ||
                error.message.includes('중단')
            )) {
                console.log('[AnalysisController] 분석 취소됨:', error.message);
                if (reportBtn) {
                    reportBtn.classList.remove('analyzing', 'completed');
                }
                return;
            }

            // 실제 오류만 로깅
            console.error('[AnalysisController] 분석 오류:', error);

            if (reportBtn) {
                reportBtn.classList.remove('analyzing', 'completed');
            }

            // 재시도 가능한 오류인지 확인
            const isRetryableError = (
                error.message.includes('Failed to fetch') ||
                error.message.includes('ERR_CONNECTION_RESET') ||
                error.message.includes('네트워크') ||
                error.message.includes('서버 연결') ||
                error.message.includes('서버 오류')
            );

            const maxRetries = this.state.maxRetries || 2;
            if (isRetryableError && retryCount < maxRetries && !this.state.isCancelled) {
                // [Genius Logic] Exponential Backoff 적용
                const backoffDelay = Math.pow(2, retryCount + 1) * 1000;
                console.warn(`[AnalysisController] 재시도 가능한 오류 감지, ${backoffDelay}ms 후 재시도...`);
                
                this.setStatus(`일시적인 연결 문제입니다. ${backoffDelay/1000}초 후 재시도합니다...`, true);
                
                await new Promise(resolve => setTimeout(resolve, backoffDelay));
                // 재시도 전 취소 플래그 재확인
                if (this.state.isCancelled) {
                    console.log('[AnalysisController] 취소됨 - 재시도 중단');
                    return;
                }
                return this.runAnalysis(text, retryCount + 1);
            }

            // 최종 실패 처리
            const errorMessage = error.message || '알 수 없는 오류가 발생했습니다.';
            this.setStatus('분석 실패: ' + errorMessage, true);
            this.showError('분석 중 오류가 발생했습니다: ' + errorMessage);
            this.showProgress({
                mode: this.state.currentMode || 'balanced',
                primary: '분석에 실패했습니다.',
                secondary: errorMessage,
                persistent: false
            });
            this.scheduleProgressClear(4000);

            // 샘플 데이터는 마지막 재시도 실패 시에만 표시
            if (retryCount >= maxRetries) {
                this.lastResult = this.buildSampleData(text);
                this.openModal();

                if (AnalysisRenderer) {
                    AnalysisRenderer.render(this.lastResult);
                    AnalysisRenderer.showError('API 연결 실패. 샘플 데이터를 표시합니다. (FastAPI 서버가 http://localhost:8000 에서 실행 중인지 확인해주세요)');
                }
            }
        } finally {
            // 취소 플래그가 설정되지 않았을 때만 상태 정리
            if (!this.state.isCancelled) {
                this.setAnalyzing(false);
            }
            this.state.abortController = null; // 정리
            this.state.currentJobId = null; // 작업 ID 정리
        }
    },

    setStatus(message, isError = false) {
        if (AnalysisRenderer) {
            AnalysisRenderer.setStatus(message, isError);
        }
    },

    showError(message) {
        if (AnalysisRenderer) {
            AnalysisRenderer.showError(message);
        }
    },

    showSample() {
        const { input, reportBtn } = this.state;
        const sampleText = DEFAULT_SAMPLE_INPUT;
        const sampleData = this.buildSampleData(sampleText, { mode: 'sample' });
        sampleData.text = sampleData.text || sampleText;
        sampleData.inputText = sampleData.inputText || sampleText;

        this.lastResult = sampleData;
        this.state.currentMode = 'sample';

        this.clearProgress(true);
        this.showProgress({
            mode: 'sample',
            primary: '샘플 프리뷰를 표시합니다.',
            secondary: '실제 분석 전에 UI 동작을 확인할 수 있는 예시 데이터입니다.',
            persistent: false
        });
        this.scheduleProgressClear(2500);

        if (input) {
            input.focus({ preventScroll: true });
        }

        this.openModal();
        this.setStatus('샘플 프리뷰 결과입니다.');
        this.setAnalyzing(false);
        if (reportBtn) {
            reportBtn.classList.remove('analyzing');
            reportBtn.classList.add('completed');
        }

        if (AnalysisRenderer) {
            AnalysisRenderer.render(sampleData);
            AnalysisRenderer.setStatus('샘플 프리뷰 결과입니다.');
            AnalysisRenderer.showError('샘플 데이터입니다. 실제 분석 결과가 아닙니다.');
        }
    },

    // 분석 단계 정의 (실제 서버 실행 순서에 맞춤)
    // 서버 로그 기반 실제 모듈 실행 순서:
    // 1. 임베딩 생성 (embedding_generation)
    // 2. 감정 강도 분석 (intensity_analysis)
    // 3. 언어 패턴 매칭 (linguistic_matcher)
    // 4. 감정 패턴 분석 (pattern_extractor)
    // 5. 맥락 추출 (context_extractor)
    // 6. 감정 관계 분석 (relationship_analyzer)
    // 7. 감정 전이 분석 (transition_analyzer)
    // 8. 최종 결과 생성
    getAnalysisSteps(mode) {
        if (mode === 'balanced') {
            return [
                { id: 'preprocessing', label: '텍스트 전처리', duration: 5, moduleName: null },
                { id: 'embedding', label: '임베딩 생성', duration: 30, moduleName: 'embedding_generation' },
                { id: 'intensity', label: '감정 강도 분석', duration: 60, moduleName: 'intensity_analysis' },
                { id: 'linguistic', label: '언어 패턴 매칭', duration: 20, moduleName: 'linguistic_matcher' },
                { id: 'pattern', label: '감정 패턴 분석', duration: 120, moduleName: 'pattern_extractor' },
                { id: 'context', label: '맥락 추출', duration: 40, moduleName: 'context_extractor' },
                { id: 'relationship', label: '감정 관계 분석', duration: 60, moduleName: 'relationship_analyzer' },
                { id: 'transition', label: '감정 전이 분석', duration: 45, moduleName: 'transition_analyzer' },
                { id: 'finalizing', label: '최종 결과 생성', duration: 20, moduleName: null }
            ];
        } else {
            return [
                { id: 'preprocessing', label: '텍스트 전처리', duration: 3, moduleName: null },
                { id: 'fast-analysis', label: '빠른 감정 분석', duration: 10, moduleName: null },
                { id: 'finalizing', label: '결과 생성', duration: 5, moduleName: null }
            ];
        }
    },
    
    // 모듈 이름을 한국어 레이블로 매핑
    getModuleLabel(moduleName) {
        const moduleLabelMap = {
            'embedding_generation': '임베딩 생성',
            'intensity_analysis': '감정 강도 분석',
            'intensity_analyzer': '감정 강도 분석',
            'linguistic_matcher': '언어 패턴 매칭',
            'pattern_extractor': '감정 패턴 분석',
            'context_extractor': '맥락 추출',
            'context_analysis': '맥락 분석',
            'relationship_analyzer': '감정 관계 분석',
            'emotion_relationship_analyzer': '감정 관계 분석',
            'transition_analyzer': '감정 전이 분석',
            'time_series_analyzer': '시계열 분석',
            'complex_analyzer': '복합 분석',
            'psychological_analyzer': '심리 분석',
            'situation_analyzer': '상황 분석',
            'weight_calculator': '가중치 계산'
        };
        return moduleLabelMap[moduleName] || moduleName;
    },

    updateProgressBar(stepIndex, totalSteps) {
        const { progressBarFill, progressBarContainer } = this.state;
        if (!progressBarFill || !progressBarContainer) return;
        
        const percentage = Math.min(95, (stepIndex / totalSteps) * 100);
        progressBarFill.style.width = `${percentage}%`;
        progressBarContainer.style.display = 'flex';

        // 미니 프로그레스 업데이트
        this.updateMiniProgress(percentage);
    },

    updateProgressSteps(currentStepIndex, steps) {
        const { progressSteps } = this.state;
        if (!progressSteps) return;

        progressSteps.innerHTML = '';
        steps.forEach((step, index) => {
            const stepEl = document.createElement('div');
            stepEl.className = 'analysis-progress__step';
            
            if (index < currentStepIndex) {
                stepEl.classList.add('completed');
                stepEl.innerHTML = `
                    <span class="analysis-progress__step-icon">
                        <svg viewBox="0 0 20 20" fill="none">
                            <path d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" fill="currentColor"/>
                        </svg>
                    </span>
                    <span>${step.label}</span>
                `;
            } else if (index === currentStepIndex) {
                stepEl.classList.add('active');
                stepEl.innerHTML = `
                    <span class="analysis-progress__step-icon">
                        <svg viewBox="0 0 20 20" fill="none">
                            <circle cx="10" cy="10" r="4" fill="currentColor" opacity="0.3"/>
                            <circle cx="10" cy="10" r="2" fill="currentColor"/>
                        </svg>
                    </span>
                    <span>${step.label}</span>
                `;
            } else {
                stepEl.innerHTML = `
                    <span class="analysis-progress__step-icon">
                        <svg viewBox="0 0 20 20" fill="none">
                            <circle cx="10" cy="10" r="3" stroke="currentColor" stroke-width="1.5" fill="none" opacity="0.3"/>
                        </svg>
                    </span>
                    <span>${step.label}</span>
                `;
            }
            
            progressSteps.appendChild(stepEl);
        });
    },

    startProgressTimer() {
        if (this.state.progressTimer) {
            clearInterval(this.state.progressTimer);
        }
        
        this.state.analysisStartTime = Date.now();
        const { progressElapsed, progressEstimated } = this.state;
        
        this.state.progressTimer = setInterval(() => {
            if (!this.state.analysisStartTime) return;
            
            const elapsed = Math.floor((Date.now() - this.state.analysisStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            
            if (progressElapsed) {
                if (minutes > 0) {
                    progressElapsed.textContent = `경과: ${minutes}분 ${seconds}초`;
                } else {
                    progressElapsed.textContent = `경과: ${seconds}초`;
                }
            }
        }, 1000);
    },

    stopProgressTimer() {
        if (this.state.progressTimer) {
            clearInterval(this.state.progressTimer);
            this.state.progressTimer = null;
        }
        this.state.analysisStartTime = null;
    },

    showProgress({ mode, primary, secondary, persistent = false, stepIndex = null, showDetailed = false } = {}) {
        const { progress, progressText, progressSub, progressBarContainer } = this.state;
        if (!progress) return;

        if (this.state.progressTimeout) {
            clearTimeout(this.state.progressTimeout);
            this.state.progressTimeout = null;
        }

        if (mode) {
            progress.dataset.mode = mode;
        } else {
            delete progress.dataset.mode;
        }

        progress.classList.add('active');
        progress.setAttribute('aria-hidden', 'false');

        if (progressText) {
            progressText.textContent = primary || '';
        }

        if (progressSub) {
            progressSub.textContent = secondary || '';
        }

        // 상세 진행 상태 표시 (BALANCED 모드이고 persistent일 때)
        if (showDetailed && mode === 'balanced' && persistent && progressBarContainer) {
            const steps = this.getAnalysisSteps(mode);
            const currentStep = stepIndex !== null ? stepIndex : this.state.currentStep;
            
            // 진행 바 업데이트
            this.updateProgressBar(currentStep, steps.length);
            
            // 단계 표시 업데이트
            this.updateProgressSteps(currentStep, steps);
            
            // 타이머 시작
            if (!this.state.progressTimer) {
                this.startProgressTimer();
            }
            
            // 단계 자동 진행 (시뮬레이션)
            if (stepIndex === null) {
                this.simulateProgressSteps(steps, mode);
            }
        } else if (progressBarContainer) {
            progressBarContainer.style.display = 'none';
            this.stopProgressTimer();
        }

        this.state.progressPersistent = !!persistent;
    },

    simulateProgressSteps(steps, mode) {
        // 실제 진행 상황을 모르므로 시간 기반으로 단계 시뮬레이션
        // 실제 서버 실행 시간을 고려하여 더 현실적인 타이밍 적용
        const totalDuration = steps.reduce((sum, step) => sum + step.duration, 0);
        const estimatedMinutes = Math.ceil(totalDuration / 60);
        
        const { progressEstimated } = this.state;
        if (progressEstimated) {
            progressEstimated.textContent = `예상: 약 ${estimatedMinutes}분`;
        }

        let currentStep = 0;
        let accumulatedTime = 0;
        
        steps.forEach((step, index) => {
            setTimeout(() => {
                if (this.state.progressPersistent && this.state.progressTimer && !this.state.isCancelled) {
                    // [FIX] UI 싱크 맞춤: 현재 처리 중인 단계를 Active로 표시 (index 사용)
                    currentStep = index;
                    this.state.currentStep = currentStep;
                    
                    this.updateProgressBar(currentStep, steps.length);
                    this.updateProgressSteps(currentStep, steps);
                    
                    // 현재 단계 메시지 업데이트
                    const { progressText, progressSub } = this.state;
                    // [FIX] 마지막 단계도 텍스트 업데이트
                    if (progressText) {
                        progressText.textContent = `${step.label} 중...`;
                    }
                    if (progressSub) {
                        progressSub.textContent = '모든 모듈이 순차적으로 실행됩니다.';
                    }
                }
            }, accumulatedTime * 1000);
            
            accumulatedTime += step.duration;
        });
    },
    
    // 서버에서 받은 module_details를 기반으로 실제 완료된 모듈 반영
    updateProgressFromModuleDetails(moduleDetails, mode) {
        if (!moduleDetails || !Array.isArray(moduleDetails)) return;
        
        const steps = this.getAnalysisSteps(mode);
        if (!steps || steps.length === 0) return;
        
        // 완료된 모듈 수 계산
        const completedModules = moduleDetails.filter(detail => 
            detail.status === 'ok' || detail.status === 'skipped'
        ).length;
        
        // 각 단계의 moduleName과 매칭하여 완료 상태 확인
        let completedSteps = 0;
        const moduleNameMap = new Map(moduleDetails.map(d => [d.name, d.status]));
        
        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            if (step.moduleName) {
                // 모듈 이름 매칭 (여러 가능한 이름 체크)
                const possibleNames = [
                    step.moduleName,
                    step.moduleName.replace('_analysis', '_analyzer'),
                    step.moduleName.replace('_analyzer', '_analysis'),
                    step.moduleName.replace('emotion_', '')
                ];
                
                let isCompleted = false;
                for (const name of possibleNames) {
                    if (moduleNameMap.has(name)) {
                        const status = moduleNameMap.get(name);
                        if (status === 'ok' || status === 'skipped') {
                            isCompleted = true;
                            break;
                        }
                    }
                }
                
                if (isCompleted) {
                    completedSteps = i + 1;
                } else {
                    break; // 첫 번째 미완료 단계에서 중단
                }
            } else {
                // moduleName이 없는 단계(전처리, 최종 결과)는 항상 완료로 간주
                completedSteps = i + 1;
            }
        }
        
        // 프로그레스 업데이트
        if (completedSteps > 0) {
            this.state.currentStep = completedSteps;
            this.updateProgressBar(completedSteps, steps.length);
            this.updateProgressSteps(completedSteps, steps);
        }
    },

    clearProgress(force = false) {
        const { progress, progressText, progressSub, progressBarContainer } = this.state;
        if (!progress) return;

        if (!force && this.state.progressPersistent) {
            return;
        }

        progress.classList.remove('active');
        progress.setAttribute('aria-hidden', 'true');
        delete progress.dataset.mode;

        if (progressText) {
            progressText.textContent = '';
        }

        if (progressSub) {
            progressSub.textContent = '';
        }

        if (progressBarContainer) {
            progressBarContainer.style.display = 'none';
        }

        this.stopProgressTimer();

        if (this.state.progressTimeout) {
            clearTimeout(this.state.progressTimeout);
            this.state.progressTimeout = null;
        }

        this.state.progressPersistent = false;
        this.state.currentStep = 0;
    },

    scheduleProgressClear(delay = 1800) {
        if (this.state.progressTimeout) {
            clearTimeout(this.state.progressTimeout);
        }

        this.state.progressPersistent = false;
        this.state.progressTimeout = window.setTimeout(() => {
            this.clearProgress(true);
            this.state.progressTimeout = null;
        }, delay);
    },

    markReportAlert() {
        const { notificationBadge, notificationBtn } = this.state;
        if (notificationBadge) notificationBadge.classList.add('active');
        if (notificationBtn) {
            notificationBtn.classList.add('has-alert');
            notificationBtn.setAttribute('aria-label', '분석 리포트 보기 (새 분석 결과가 있습니다)');
        }
    },

    clearReportAlert() {
        const { notificationBadge, notificationBtn } = this.state;
        if (notificationBadge) notificationBadge.classList.remove('active');
        if (notificationBtn) {
            notificationBtn.classList.remove('has-alert');
            notificationBtn.setAttribute('aria-label', '분석 리포트 보기');
        }
    },

    buildSampleData(text, options = {}) {
        const { mode = 'sample' } = options || {};
        const normalizedMode = typeof mode === 'string' ? mode.toLowerCase() : 'sample';
        return {
            success: true,
            text: text,
            inputText: text,
            meta: {
                mode: normalizedMode,
                elapsed: 0.123,
                timestamp: new Date().toISOString(),
                evidence_score: 0.85,
                sample_preview: true
            },
            main_distribution: {
                '희': 0.15,
                '노': 0.45,
                '애': 0.30,
                '락': 0.10
            },
            products: {
                p1: {
                    headline_emotions: ['노', '애'],
                    intensity: '중상',
                    triggers: ['보험료 인상', '사전 안내 부족'],
                    recommended_actions: ['즉시 상담', '혜택 재설명']
                }
            },
            insight_summary: [
                `입력 텍스트: ${text.slice(0, 50)}...`,
                '주요 감정: 노(분노) 45%, 애(슬픔) 30%',
                '핵심 트리거: 비용 부담, 소통 부재',
                '권장 조치: 즉각적인 개입 필요'
            ],
            master_report: '== SAMPLE DATA ==\n분석 결과 샘플입니다.\nAPI 서버가 연결되지 않았습니다.'
        };
    }
};

const App = {
    init() {
        VideoController.init();
        Slider.init();
        Modal.init();
        Tabs.init();
        MobileMenu.init();
        DemoFloat.init();
        AnalysisController.init();
    }
};

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => App.init());
} else {
    App.init();
}

